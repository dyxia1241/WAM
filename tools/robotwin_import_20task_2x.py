#!/usr/bin/env python3
"""Import RoboTwin 20-task 2x rule-labeled variants into WAM episodes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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
    parser.add_argument("--output", default="data/episodes/robotwin_20task_2x_rule_subsuccess_v1")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-images", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    wam_root = Path(args.wam_root)
    sys.path.insert(0, str(wam_root))
    from ppwam.import_robotwin import import_one_hdf5

    robotwin_root = Path(args.robotwin_root)
    output = wam_root / args.output
    sidecar_dir = wam_root / args.sidecar_dir
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for task in args.tasks:
        for kind in KINDS:
            variant = f"ppwam_{kind}_{args.variant_suffix}"
            hdf5 = robotwin_root / "data" / task / variant / "data" / "episode0.hdf5"
            sidecar = sidecar_dir / f"{task}_{kind}_{args.variant_suffix}_rule_v1_sidecar.json"
            instructions = robotwin_root / "data" / task / variant / "instructions" / "episode0.json"
            scene_info = robotwin_root / "data" / task / variant / "scene_info.json"
            if not hdf5.exists():
                rows.append({"task": task, "variant": variant, "status": "missing_hdf5", "hdf5": str(hdf5)})
                continue
            if not sidecar.exists():
                rows.append({"task": task, "variant": variant, "status": "missing_sidecar", "sidecar": str(sidecar)})
                continue
            try:
                imported = import_one_hdf5(
                    hdf5_path=hdf5,
                    output_root=output,
                    task_id=task,
                    instructions=instructions if instructions.exists() else "",
                    scene_info=scene_info if scene_info.exists() else "",
                    label_sidecar=sidecar,
                    episode_id_prefix=f"{task}_{kind}_{args.variant_suffix}",
                    skip_images=args.skip_images,
                    overwrite=args.overwrite,
                )
                rows.append(
                    {
                        "task": task,
                        "variant": variant,
                        "status": "ok",
                        "episode_id": imported.episode_id,
                        "frames": imported.num_frames,
                        "sidecar": str(sidecar),
                    }
                )
            except Exception as exc:
                rows.append({"task": task, "variant": variant, "status": "failed", "error": repr(exc)})
    manifest = {"num_ok": sum(1 for row in rows if row["status"] == "ok"), "num_total": len(rows), "episodes": rows}
    (output / "robotwin_import_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({"output": str(output), "num_ok": manifest["num_ok"], "num_total": manifest["num_total"]}, indent=2))
    return 0 if manifest["num_ok"] == manifest["num_total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
