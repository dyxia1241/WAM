#!/usr/bin/env python3
"""Batch-generate RoboTwin 2x perturbation replay variants.

This script is intended to run on the 5060 machine. It operates on a RoboTwin
checkout and uses RoboTwin's own collect_data.py for planning/replay.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

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
    parser.add_argument("--python", default="/home/dayu/anaconda3/envs/RoboTwin/bin/python")
    parser.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    parser.add_argument("--kinds", nargs="*", choices=KINDS, default=list(KINDS))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--clean-config", default="ppwam_smoke_clean")
    parser.add_argument("--variant-suffix", default="2x_v1")
    parser.add_argument("--insert-rows", type=int, default=1200)
    parser.add_argument("--detour-amp", type=float, default=0.10)
    parser.add_argument("--overshoot-amp", type=float, default=0.10)
    parser.add_argument("--force-clean", action="store_true")
    parser.add_argument("--force-variants", action="store_true")
    parser.add_argument(
        "--hesitation-policy",
        choices=("second_longest", "final_segment"),
        default="second_longest",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(cmd: list[str], cwd: Path, env: dict[str, str], log_path: Path, dry_run: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUN", " ".join(cmd), "cwd=", cwd, "log=", log_path, flush=True)
    if dry_run:
        return 0
    with log_path.open("ab") as log:
        log.write(("\n\n===== " + time.strftime("%Y-%m-%d %H:%M:%S") + " =====\n").encode())
        log.write((" ".join(cmd) + "\n").encode())
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    return int(proc.returncode)


def ensure_variant_configs(root: Path, clean_config: str, variant_suffix: str) -> None:
    src = root / "task_config" / f"{clean_config}.yml"
    if not src.exists():
        raise FileNotFoundError(src)
    text = src.read_text()
    text = text.replace("use_seed: false", "use_seed: true")
    text = text.replace("episode_num: 50", "episode_num: 1")
    text = text.replace("episode_num: 3", "episode_num: 1")
    text = text.replace("episode_num: 2", "episode_num: 1")
    text = text.replace("clear_cache_freq: 5", "clear_cache_freq: 1")
    for kind in KINDS:
        dst = root / "task_config" / f"ppwam_{kind}_{variant_suffix}.yml"
        if not dst.exists() or dst.read_text() != text:
            dst.write_text(text)


def hdf5_done(root: Path, task: str, config: str) -> bool:
    return (root / "data" / task / config / "data" / "episode0.hdf5").exists()


def pkl_done(root: Path, task: str, config: str) -> bool:
    return (root / "data" / task / config / "_traj_data" / "episode0.pkl").exists()


def load_traj(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Unexpected trajectory object in {path}: {type(obj)}")
    return obj


def dump_traj(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(obj, f)


def segment_array(segment: dict[str, Any]) -> np.ndarray:
    arr = np.asarray(segment["position"], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[0] < 4:
        raise ValueError(f"Bad segment position shape: {arr.shape}")
    return arr


def active_segments(traj: dict[str, Any]) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for arm_key in ("left_joint_path", "right_joint_path"):
        for idx, seg in enumerate(traj.get(arm_key, [])):
            if isinstance(seg, dict) and "position" in seg:
                out.append((arm_key, idx, int(np.asarray(seg["position"]).shape[0])))
    return out


def choose_segment(traj: dict[str, Any], kind: str, hesitation_policy: str = "second_longest") -> tuple[str, int]:
    segs = active_segments(traj)
    if not segs:
        raise ValueError("No editable joint-path segments found.")
    segs_sorted = sorted(segs, key=lambda x: x[2], reverse=True)
    if kind == "detour":
        return segs_sorted[0][0], segs_sorted[0][1]
    if kind == "hesitation":
        if hesitation_policy == "final_segment":
            arm_keys = [key for key in ("left_joint_path", "right_joint_path") if traj.get(key)]
            if arm_keys:
                arm_key = max(arm_keys, key=lambda key: sum(int(np.asarray(s["position"]).shape[0]) for s in traj[key]))
                return arm_key, len(traj[arm_key]) - 1
        return segs_sorted[min(1, len(segs_sorted) - 1)][0], segs_sorted[min(1, len(segs_sorted) - 1)][1]
    return segs_sorted[-1][0], segs_sorted[-1][1]


def recompute_velocity(pos: np.ndarray) -> np.ndarray:
    if len(pos) <= 1:
        return np.zeros_like(pos, dtype=np.float32)
    vel = np.zeros_like(pos, dtype=np.float32)
    vel[1:] = pos[1:] - pos[:-1]
    vel[0] = vel[1]
    return vel


def replace_segment(segment: dict[str, Any], pos: np.ndarray) -> dict[str, Any]:
    updated = dict(segment)
    updated["position"] = pos.astype(np.float32)
    updated["velocity"] = recompute_velocity(updated["position"])
    return updated


def smooth_loop(base: np.ndarray, offset: np.ndarray, rows: int) -> np.ndarray:
    phase = np.linspace(0.0, 2.0 * np.pi, rows, endpoint=False, dtype=np.float32)
    return base[None, :] + np.sin(phase)[:, None] * offset[None, :]


def edit_traj(
    traj: dict[str, Any],
    kind: str,
    insert_rows: int,
    detour_amp: float,
    overshoot_amp: float,
    hesitation_policy: str = "second_longest",
) -> tuple[dict[str, Any], dict[str, Any]]:
    arm_key, seg_idx = choose_segment(traj, kind, hesitation_policy=hesitation_policy)
    seg = traj[arm_key][seg_idx]
    pos = segment_array(seg)
    n = int(pos.shape[0])
    ratio = 0.55 if kind != "detour" else 0.45
    if kind == "hesitation" and hesitation_policy == "final_segment":
        ratio = 0.85
    insert_at = max(2, min(n - 2, int(round(n * ratio))))
    base = pos[insert_at].astype(np.float32)

    if kind == "hesitation":
        insert = np.repeat(base[None, :], insert_rows, axis=0)
    elif kind == "detour":
        direction = np.zeros(pos.shape[1], dtype=np.float32)
        direction[: min(6, pos.shape[1])] = np.asarray([1.0, -0.5, 0.8, -0.4, 0.2, 0.8], dtype=np.float32)[
            : min(6, pos.shape[1])
        ]
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        insert = smooth_loop(base, direction * float(detour_amp), insert_rows)
    elif kind == "overshoot":
        if n >= 4:
            direction = pos[-1] - pos[max(0, n - 12)]
        else:
            direction = pos[-1] - pos[0]
        if float(np.linalg.norm(direction)) < 1e-6:
            direction = np.ones(pos.shape[1], dtype=np.float32)
        direction = direction.astype(np.float32)
        direction /= max(float(np.linalg.norm(direction)), 1e-6)
        insert = smooth_loop(base, direction * float(overshoot_amp), insert_rows)
    else:
        raise ValueError(kind)

    new_pos = np.concatenate([pos[: insert_at + 1], insert.astype(np.float32), pos[insert_at + 1 :]], axis=0)
    edited = {k: list(v) if isinstance(v, list) else v for k, v in traj.items()}
    edited[arm_key] = list(traj[arm_key])
    edited[arm_key][seg_idx] = replace_segment(seg, new_pos)
    summary = {
        "kind": kind,
        "edited_arm": arm_key,
        "edited_segment": seg_idx,
        "source_lengths": [length for a, _i, length in active_segments(traj) if a == arm_key],
        "edited_lengths": [int(np.asarray(s["position"]).shape[0]) for s in edited[arm_key]],
        "insert_at": insert_at,
        "insert_rows": int(insert_rows),
        "segment_original_rows": n,
        "segment_new_rows": int(new_pos.shape[0]),
    }
    return edited, summary


def prepare_variant(root: Path, task: str, clean_config: str, variant: str, kind: str, args: argparse.Namespace) -> None:
    clean_dir = root / "data" / task / clean_config
    dst_dir = root / "data" / task / variant
    src_pkl = clean_dir / "_traj_data" / "episode0.pkl"
    if not src_pkl.exists():
        raise FileNotFoundError(src_pkl)
    if dst_dir.exists() and args.force_variants:
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for name in ("seed.txt",):
        src = clean_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)
    traj = load_traj(src_pkl)
    edited, summary = edit_traj(
        traj,
        kind=kind,
        insert_rows=args.insert_rows,
        detour_amp=args.detour_amp,
        overshoot_amp=args.overshoot_amp,
        hesitation_policy=args.hesitation_policy,
    )
    summary.update({"task": task, "variant": variant, "source_config": clean_config})
    dump_traj(dst_dir / "_traj_data" / "episode0.pkl", edited)
    (dst_dir / f"{kind}_{args.variant_suffix}_edit_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def main() -> int:
    args = parse_args()
    root = Path(args.robotwin_root)
    if not (root / "script" / "collect_data.py").exists():
        raise FileNotFoundError(root / "script" / "collect_data.py")
    ensure_variant_configs(root, args.clean_config, args.variant_suffix)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env["PYTHONWARNINGS"] = "ignore::UserWarning"
    log_dir = root / "logs" / "ppwam_20task_2x"
    manifest: list[dict[str, Any]] = []

    for task in args.tasks:
        clean_done = hdf5_done(root, task, args.clean_config) and pkl_done(root, task, args.clean_config)
        if args.force_clean or not clean_done:
            code = run(
                [args.python, "script/collect_data.py", task, args.clean_config],
                cwd=root,
                env=env,
                log_path=log_dir / f"{task}_{args.clean_config}.log",
                dry_run=args.dry_run,
            )
            if code != 0:
                manifest.append({"task": task, "stage": "clean", "status": "failed", "returncode": code})
                continue
        for kind in tuple(args.kinds):
            variant = f"ppwam_{kind}_{args.variant_suffix}"
            if not hdf5_done(root, task, variant) or args.force_variants:
                if not args.dry_run:
                    prepare_variant(root, task, args.clean_config, variant, kind, args)
                code = run(
                    [args.python, "script/collect_data.py", task, variant],
                    cwd=root,
                    env=env,
                    log_path=log_dir / f"{task}_{variant}.log",
                    dry_run=args.dry_run,
                )
                status = "ok" if code == 0 and (args.dry_run or hdf5_done(root, task, variant)) else "failed"
            else:
                code = 0
                status = "exists"
            manifest.append({"task": task, "variant": variant, "kind": kind, "status": status, "returncode": code})
        manifest_path = log_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
