#!/usr/bin/env python3
"""Generate task-specific RoboTwin potential sidecars for the 2-task pilot."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


TASKS = ("beat_block_hammer", "click_bell")
KINDS = ("hesitation", "detour", "overshoot")


@dataclass(frozen=True)
class TaskRuleProfile:
    active_arm: str
    arc_weight: float
    target_weight: float
    projection_weight: float
    detour_penalty: float
    overshoot_penalty: float


TASK_PROFILES = {
    "click_bell": TaskRuleProfile(
        active_arm="left",
        arc_weight=0.35,
        target_weight=0.55,
        projection_weight=0.10,
        detour_penalty=0.10,
        overshoot_penalty=0.16,
    ),
    "beat_block_hammer": TaskRuleProfile(
        active_arm="right",
        arc_weight=0.45,
        target_weight=0.35,
        projection_weight=0.20,
        detour_penalty=0.12,
        overshoot_penalty=0.20,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", default="/data/projects/RoboTwin")
    parser.add_argument("--wam-root", default="/data/projects/WAM")
    parser.add_argument("--tasks", nargs="*", default=list(TASKS))
    parser.add_argument("--variant-suffix", default="2x_v1")
    parser.add_argument("--sidecar-dir", default="data/robotwin_sidecars")
    parser.add_argument("--figure-dir", default="docs/figures/current/robotwin_2task_rule_v2_potential")
    parser.add_argument("--include-expert", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    loaded = json.loads(path.read_text())
    return loaded if isinstance(loaded, dict) else {}


def hdf5_frame_count(path: Path) -> int:
    with h5py.File(path, "r") as h5:
        if "joint_action/vector" in h5:
            return int(h5["joint_action/vector"].shape[0])
        for camera in ("head_camera", "front_camera", "left_camera", "right_camera"):
            key = f"observation/{camera}/rgb"
            if key in h5:
                return int(h5[key].shape[0])
    raise ValueError(f"Cannot infer frame count from {path}")


def active_xyz(path: Path, active_arm: str) -> np.ndarray:
    key = f"endpose/{active_arm}_endpose"
    with h5py.File(path, "r") as h5:
        if key not in h5:
            raise ValueError(f"Missing active-arm endpose dataset {key} in {path}")
        arr = np.asarray(h5[key][()], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Bad endpose shape for {key}: {arr.shape}")
    return arr[:, :3].astype(np.float32)


def instruction_for(root: Path, task: str, variant: str) -> str:
    p = root / "data" / task / variant / "instructions" / "episode0.json"
    if not p.exists():
        return ""
    data = read_json_if_exists(p)
    for key in ("seen", "unseen"):
        value = data.get(key)
        if isinstance(value, list) and value and isinstance(value[0], str):
            return value[0]
    for key in ("instruction", "language", "task_description"):
        if isinstance(data.get(key), str):
            return data[key]
    return ""


def find_summary(variant_dir: Path, kind: str) -> dict[str, Any]:
    candidates = sorted(variant_dir.glob(f"*{kind}*summary.json")) + sorted(variant_dir.glob("*summary.json"))
    if not candidates:
        return {}
    return read_json_if_exists(candidates[0])


def dense_interval(summary: dict[str, Any], kind: str) -> tuple[float, float]:
    insert_rows = int(summary.get("insert_rows", summary.get("inserted_rows", summary.get("appended_rows", 1200))))
    insert_at = int(summary.get("insert_at", summary.get("insert_after_row", 0)))
    edited_segment = int(summary.get("edited_segment", 0))
    source_lengths = summary.get("source_lengths") or []
    if not source_lengths:
        original = int(summary.get("segment_original_rows", max(insert_at + 2, 100)))
        source_lengths = [original]
        edited_segment = 0
    before = int(sum(int(x) for x in source_lengths[:edited_segment])) + insert_at
    total = int(sum(int(x) for x in source_lengths)) + insert_rows
    if kind == "overshoot" and "appended_rows" in summary:
        before = max(0, int(sum(int(x) for x in source_lengths[: edited_segment + 1])) - 1)
    start = max(0.0, min(0.98, before / max(total - 1, 1)))
    end = max(start + 0.02, min(0.99, (before + insert_rows) / max(total - 1, 1)))
    return start, end


def normalize_01(values: np.ndarray) -> np.ndarray:
    lo = float(np.nanmin(values)) if values.size else 0.0
    hi = float(np.nanmax(values)) if values.size else 1.0
    if hi - lo < 1.0e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def arc_progress(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if xyz.shape[0] <= 1:
        return np.ones((xyz.shape[0],), dtype=np.float32), np.zeros((xyz.shape[0],), dtype=np.float32)
    step = np.linalg.norm(np.diff(xyz, axis=0), axis=1).astype(np.float32)
    arc = np.concatenate([[0.0], np.cumsum(step)])
    if float(arc[-1]) <= 1.0e-6:
        return np.linspace(0.0, 1.0, xyz.shape[0], dtype=np.float32), np.concatenate([[0.0], step])
    return (arc / float(arc[-1])).astype(np.float32), np.concatenate([[0.0], step]).astype(np.float32)


def target_progress(xyz: np.ndarray) -> np.ndarray:
    final = xyz[-1]
    dist = np.linalg.norm(xyz - final[None, :], axis=1).astype(np.float32)
    denom = max(float(np.max(dist)), 1.0e-6)
    return np.clip(1.0 - dist / denom, 0.0, 1.0).astype(np.float32)


def projection_progress(xyz: np.ndarray) -> np.ndarray:
    start = xyz[0]
    final = xyz[-1]
    direction = final - start
    denom = max(float(np.dot(direction, direction)), 1.0e-6)
    proj = ((xyz - start[None, :]) @ direction) / denom
    return np.clip(proj, 0.0, 1.0).astype(np.float32)


def interval_mask(n: int, start_frac: float, end_frac: float) -> np.ndarray:
    last = max(n - 1, 1)
    s = max(0, min(n - 1, int(round(start_frac * last))))
    e = max(s + 1, min(n - 1, int(round(end_frac * last))))
    mask = np.zeros((n,), dtype=np.float32)
    if e <= s:
        return mask
    mid = (s + e) / 2.0
    for i in range(s, e + 1):
        width = max(mid - s, e - mid, 1.0)
        mask[i] = max(0.0, 1.0 - abs(float(i) - mid) / width)
    return mask


def success_rescale(phi: np.ndarray, monotonic: bool = False) -> np.ndarray:
    phi = np.clip(phi.astype(np.float32), 0.0, 1.0)
    phi = phi - float(phi[0])
    hi = max(float(np.max(phi)), 1.0e-6)
    phi = np.clip(phi / hi, 0.0, 1.0)
    if monotonic:
        phi = np.maximum.accumulate(phi)
    phi[0] = 0.0
    phi[-1] = 1.0
    return phi.astype(np.float32)


def potential_for(
    hdf5_path: Path,
    task: str,
    kind: str,
    edit_summary: dict[str, Any],
) -> tuple[list[float], dict[str, Any]]:
    profile = TASK_PROFILES[task]
    xyz = active_xyz(hdf5_path, profile.active_arm)
    arc, speed = arc_progress(xyz)
    target = target_progress(xyz)
    proj = projection_progress(xyz)
    phi = (
        profile.arc_weight * arc
        + profile.target_weight * target
        + profile.projection_weight * proj
    ).astype(np.float32)
    penalty = np.zeros_like(phi, dtype=np.float32)
    dense = None
    if kind in KINDS:
        start_frac, end_frac = dense_interval(edit_summary, kind)
        dense = [float(start_frac), float(end_frac)]
        mask = interval_mask(int(phi.shape[0]), start_frac, end_frac)
        if kind == "hesitation":
            s = int(round(start_frac * max(phi.shape[0] - 1, 1)))
            e = int(round(end_frac * max(phi.shape[0] - 1, 1)))
            e = max(s + 1, min(phi.shape[0] - 1, e))
            phi[s : e + 1] = np.minimum(phi[s : e + 1], phi[s])
        elif kind == "detour":
            penalty = profile.detour_penalty * mask
        elif kind == "overshoot":
            penalty = profile.overshoot_penalty * mask
        phi = phi - penalty
    phi = success_rescale(phi, monotonic=kind in {"expert", "hesitation"})
    diagnostics = {
        "rule_version": f"{task}_privileged_eef_rule_v2",
        "active_arm": profile.active_arm,
        "component_weights": {
            "arc": profile.arc_weight,
            "target": profile.target_weight,
            "projection": profile.projection_weight,
        },
        "dense_insert_interval": dense,
        "active_eef_speed": [float(x) for x in speed],
        "arc_progress": [float(x) for x in arc],
        "target_progress": [float(x) for x in target],
        "projection_progress": [float(x) for x in proj],
        "rule_penalty": [float(x) for x in penalty],
    }
    return [float(x) for x in phi], diagnostics


def write_plot(path: Path, task: str, variant: str, phi: list[float], diagnostics: dict[str, Any]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.arange(len(phi))
    plt.figure(figsize=(8, 3.2))
    plt.plot(xs, phi, linewidth=2, label="Phi")
    penalty = diagnostics.get("rule_penalty")
    if isinstance(penalty, list) and any(abs(float(x)) > 1.0e-8 for x in penalty):
        plt.plot(xs, penalty, linewidth=1, alpha=0.65, label="rule penalty")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("frame")
    plt.ylabel("potential Phi")
    plt.title(f"{task} / {variant} / rule-v2")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def write_sidecar(
    out: Path,
    hdf5_path: Path,
    robotwin_root: Path,
    task: str,
    variant: str,
    kind: str,
    edit_summary: dict[str, Any],
    figure_path: Path,
) -> dict[str, Any]:
    n = hdf5_frame_count(hdf5_path)
    phi, diagnostics = potential_for(hdf5_path, task, kind, edit_summary)
    sidecar = {
        "label_source": f"robotwin_{task}_{kind}_task_rule_v2",
        "potential_label_source": "task_specific_privileged_eef_rule_v2",
        "task_id": task,
        "variant_id": variant,
        "suboptimal_type": kind,
        "success": True,
        "language_instruction": instruction_for(robotwin_root, task, variant),
        "primitive_boundaries": [{"stage": "move", "start": 0, "end": n - 1}],
        "primitive_metadata": {
            **diagnostics,
            "edit_summary": edit_summary,
            "figure": str(figure_path),
        },
        "diagnostics": diagnostics,
        "potential": phi,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sidecar, indent=2, sort_keys=True))
    write_plot(figure_path, task, variant, phi, diagnostics)
    return {
        "task": task,
        "variant": variant,
        "kind": kind,
        "frames": n,
        "status": "ok",
        "sidecar": str(out),
        "figure": str(figure_path),
        "rule_version": diagnostics["rule_version"],
        "delta_phi_min": float(np.min(np.diff(np.asarray(phi, dtype=np.float32)))) if len(phi) > 1 else 0.0,
    }


def expert_hdf5(robotwin_root: Path, task: str) -> Path:
    return (
        robotwin_root
        / "data_archive"
        / "2026_07_14_robotwin_1x_smoke"
        / task
        / "ppwam_smoke_clean"
        / "data"
        / "episode0.hdf5"
    )


def main() -> int:
    args = parse_args()
    robotwin_root = Path(args.robotwin_root)
    wam_root = Path(args.wam_root)
    sidecar_dir = wam_root / args.sidecar_dir
    figure_dir = wam_root / args.figure_dir
    rows: list[dict[str, Any]] = []
    for task in args.tasks:
        if task not in TASK_PROFILES:
            rows.append({"task": task, "status": "unsupported_task"})
            continue
        if args.include_expert:
            hdf5 = expert_hdf5(robotwin_root, task)
            out = sidecar_dir / f"{task}_expert_direct_rule_v2_sidecar.json"
            fig = figure_dir / f"{task}_expert_direct_rule_v2_phi.png"
            if hdf5.exists() and (args.overwrite or not out.exists()):
                rows.append(write_sidecar(out, hdf5, robotwin_root, task, "expert_direct", "expert", {}, fig))
            elif hdf5.exists():
                rows.append({"task": task, "variant": "expert_direct", "kind": "expert", "status": "exists", "sidecar": str(out)})
            else:
                rows.append({"task": task, "variant": "expert_direct", "kind": "expert", "status": "missing_hdf5", "hdf5": str(hdf5)})
        for kind in KINDS:
            variant = f"ppwam_{kind}_{args.variant_suffix}"
            variant_dir = robotwin_root / "data" / task / variant
            hdf5 = variant_dir / "data" / "episode0.hdf5"
            out = sidecar_dir / f"{task}_{kind}_{args.variant_suffix}_rule_v2_sidecar.json"
            fig = figure_dir / f"{task}_{kind}_{args.variant_suffix}_rule_v2_phi.png"
            if not hdf5.exists():
                rows.append({"task": task, "variant": variant, "kind": kind, "status": "missing_hdf5", "hdf5": str(hdf5)})
                continue
            if out.exists() and not args.overwrite:
                rows.append({"task": task, "variant": variant, "kind": kind, "status": "exists", "sidecar": str(out)})
                continue
            rows.append(write_sidecar(out, hdf5, robotwin_root, task, variant, kind, find_summary(variant_dir, kind), fig))
    figure_dir.mkdir(parents=True, exist_ok=True)
    (figure_dir / "robotwin_2task_rule_v2_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    ok = sum(1 for row in rows if row.get("status") in {"ok", "exists"})
    print(json.dumps({"num_ok": ok, "num_total": len(rows), "summary": str(figure_dir / "robotwin_2task_rule_v2_summary.json")}, indent=2))
    return 0 if ok == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
