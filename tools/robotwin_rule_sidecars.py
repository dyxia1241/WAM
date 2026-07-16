#!/usr/bin/env python3
"""Generate rule-based potential sidecars for RoboTwin 2x variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_TASKS = [
    "beat_block_hammer",
    "click_bell",
    "click_alarmclock",
    "press_stapler",
    "turn_switch",
    "stamp_seal",
    "open_laptop",
    "open_microwave",
    "move_pillbottle_pad",
    "move_stapler_pad",
    "move_can_pot",
    "place_empty_cup",
    "place_shoe",
    "place_container_plate",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "stack_blocks_two",
    "stack_bowls_two",
    "handover_block",
]

KINDS = ("hesitation", "detour", "overshoot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--robotwin-root", default="/data/projects/RoboTwin")
    parser.add_argument("--wam-root", default="/data/projects/WAM")
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--variant-suffix", default="2x_v1")
    parser.add_argument("--sidecar-dir", default="data/robotwin_sidecars")
    parser.add_argument("--figure-dir", default="docs/figures/current/robotwin_20task_2x_rule_potential")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def hdf5_frame_count(path: Path) -> int:
    with h5py.File(path, "r") as h5:
        if "joint_action/vector" in h5:
            return int(h5["joint_action/vector"].shape[0])
        for camera in ("head_camera", "front_camera", "left_camera", "right_camera"):
            key = f"observation/{camera}/rgb"
            if key in h5:
                return int(h5[key].shape[0])
    raise ValueError(f"Cannot infer frame count from {path}")


def find_summary(variant_dir: Path, kind: str) -> dict[str, Any]:
    candidates = sorted(variant_dir.glob(f"*{kind}*summary.json")) + sorted(variant_dir.glob("*summary.json"))
    if not candidates:
        return {}
    return json.loads(candidates[0].read_text())


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


def interp_potential(n: int, anchors: list[tuple[int, float]]) -> list[float]:
    anchors = sorted((max(0, min(n - 1, int(x))), float(y)) for x, y in anchors)
    xs = np.asarray([x for x, _ in anchors], dtype=np.float32)
    ys = np.asarray([y for _, y in anchors], dtype=np.float32)
    frames = np.arange(n, dtype=np.float32)
    phi = np.interp(frames, xs, ys).astype(np.float32)
    phi = np.clip(phi, 0.0, 1.0)
    phi[0] = 0.0
    phi[-1] = 1.0
    return [float(x) for x in phi]


def anchors_for(kind: str, n: int, start_frac: float, end_frac: float) -> list[tuple[int, float]]:
    last = n - 1
    s = max(1, min(last - 2, int(round(start_frac * last))))
    e = max(s + 1, min(last - 1, int(round(end_frac * last))))
    mid = max(s + 1, min(e - 1, (s + e) // 2))
    base_s = s / max(last, 1)
    base_e = e / max(last, 1)
    if kind == "hesitation":
        hold = max(0.0, min(0.95, base_s))
        return [(0, 0.0), (s, hold), (e, hold), (last, 1.0)]
    if kind == "detour":
        dip = max(0.0, base_s - max(0.08, 0.35 * max(base_e - base_s, 0.02)))
        recover = min(0.95, max(base_e, base_s + 0.10))
        return [(0, 0.0), (s, base_s), (mid, dip), (e, recover), (last, 1.0)]
    if kind == "overshoot":
        dip = max(0.0, base_s - max(0.10, 0.45 * max(base_e - base_s, 0.02)))
        recover = min(0.98, max(base_e, base_s + 0.18))
        return [(0, 0.0), (s, base_s), (mid, dip), (e, recover), (last, 1.0)]
    raise ValueError(kind)


def instruction_for(robotwin_root: Path, task: str, variant: str) -> str:
    p = robotwin_root / "data" / task / variant / "instructions" / "episode0.json"
    if not p.exists():
        return ""
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return ""
    if isinstance(data, dict):
        for key in ("instruction", "language", "task_description"):
            if isinstance(data.get(key), str):
                return data[key]
        for value in data.values():
            if isinstance(value, str):
                return value
            if isinstance(value, list) and value and isinstance(value[0], str):
                return value[0]
    if isinstance(data, list) and data and isinstance(data[0], str):
        return data[0]
    return ""


def write_plot(path: Path, task: str, variant: str, phi: list[float], anchors: list[tuple[int, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    xs = np.arange(len(phi))
    plt.figure(figsize=(8, 3.2))
    plt.plot(xs, phi, linewidth=2)
    ax = plt.gca()
    for x, y in anchors:
        ax.scatter([x], [y], s=18)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("frame")
    plt.ylabel("potential Phi")
    plt.title(f"{task} / {variant}")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> int:
    args = parse_args()
    robotwin_root = Path(args.robotwin_root)
    wam_root = Path(args.wam_root)
    sidecar_dir = wam_root / args.sidecar_dir
    figure_dir = wam_root / args.figure_dir
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for task in args.tasks:
        for kind in KINDS:
            variant = f"ppwam_{kind}_{args.variant_suffix}"
            variant_dir = robotwin_root / "data" / task / variant
            hdf5_path = variant_dir / "data" / "episode0.hdf5"
            if not hdf5_path.exists():
                summary_rows.append({"task": task, "variant": variant, "status": "missing_hdf5"})
                continue
            out = sidecar_dir / f"{task}_{kind}_{args.variant_suffix}_rule_v1_sidecar.json"
            if out.exists() and not args.overwrite:
                summary_rows.append({"task": task, "variant": variant, "status": "exists", "sidecar": str(out)})
                continue
            n = hdf5_frame_count(hdf5_path)
            edit_summary = find_summary(variant_dir, kind)
            start_frac, end_frac = dense_interval(edit_summary, kind)
            anchors = anchors_for(kind, n, start_frac, end_frac)
            phi = interp_potential(n, anchors)
            sidecar = {
                "label_source": f"robotwin_subsuccess_{task}_{kind}_{args.variant_suffix}_rule_phi",
                "task_id": task,
                "variant_id": variant,
                "suboptimal_type": kind,
                "success": True,
                "language_instruction": instruction_for(robotwin_root, task, variant),
                "primitive_boundaries": [{"stage": "move", "start": 0, "end": n - 1}],
                "primitive_metadata": {
                    "rule_version": "rule_v1_generic_2x",
                    "anchors": [{"frame": int(x), "phi": float(y)} for x, y in anchors],
                    "dense_insert_interval": [float(start_frac), float(end_frac)],
                    "edit_summary": edit_summary,
                },
                "potential": phi,
            }
            out.write_text(json.dumps(sidecar, indent=2, sort_keys=True))
            fig = figure_dir / f"{task}_{kind}_{args.variant_suffix}_rule_phi.png"
            write_plot(fig, task, variant, phi, anchors)
            summary_rows.append(
                {
                    "task": task,
                    "variant": variant,
                    "status": "ok",
                    "frames": n,
                    "sidecar": str(out),
                    "figure": str(fig),
                    "anchors": anchors,
                }
            )
    (figure_dir / "robotwin_20task_2x_rule_sidecar_summary.json").write_text(
        json.dumps(summary_rows, indent=2, sort_keys=True)
    )
    ok = sum(1 for row in summary_rows if row["status"] in {"ok", "exists"})
    missing = [row for row in summary_rows if row["status"] not in {"ok", "exists"}]
    print(f"sidecars ok/existing: {ok}/{len(summary_rows)}")
    if missing:
        print("missing/failed:")
        for row in missing:
            print(row)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
