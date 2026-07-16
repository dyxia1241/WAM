from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


VARIANTS = ("hesitation", "detour", "overshoot")


def read_windows(path: str | Path) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                loaded = json.loads(line)
                if not isinstance(loaded, dict):
                    raise ValueError(f"Expected JSON object in {path}.")
                windows.append(loaded)
    if not windows:
        raise ValueError(f"No windows found in {path}.")
    return windows


def infer_variant(episode_id: str) -> str:
    if "expert_direct" in episode_id:
        return "expert"
    for variant in VARIANTS:
        if variant in episode_id:
            return variant
    return "unknown"


def infer_task(episode_id: str, task_id: str | int | None = None) -> str:
    if task_id is not None:
        raw = str(task_id)
        if not raw.isdigit():
            return raw
    for variant in VARIANTS:
        marker = f"_{variant}_"
        if marker in episode_id:
            return episode_id.split(marker, 1)[0]
    if "_expert_direct_" in episode_id:
        return episode_id.split("_expert_direct_", 1)[0]
    return "unknown"


def summarize(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        raise ValueError("Cannot summarize empty values.")
    return {
        "num_windows": float(values.size),
        "delta_phi_raw_min": float(values.min()),
        "delta_phi_raw_mean": float(values.mean()),
        "delta_phi_raw_max": float(values.max()),
        "delta_phi_raw_std": float(values.std()),
        "negative_rate": float(np.mean(values < 0.0)),
        "stagnation_rate": float(np.mean(np.isclose(values, 0.0, atol=1.0e-6))),
        "positive_rate": float(np.mean(values > 1.0e-6)),
    }


def audit_robotwin_variants(windows_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, float]]:
    windows_dir = Path(windows_dir)
    windows = read_windows(windows_dir / "windows.jsonl")
    with np.load(windows_dir / "labels.npz") as labels:
        if "delta_phi_raw" not in labels:
            raise ValueError(f"{windows_dir / 'labels.npz'} is missing delta_phi_raw.")
        raw = labels["delta_phi_raw"].astype(np.float32)

    if raw.shape[0] != len(windows):
        raise ValueError("labels.npz delta_phi_raw length does not match windows.jsonl.")

    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for index, window in enumerate(windows):
        episode_id = str(window["episode_id"])
        split = str(window.get("split", "unknown"))
        task = infer_task(episode_id, window.get("task_id"))
        variant = infer_variant(episode_id)
        groups[(split, task, variant)].append(float(raw[index]))
        groups[("all", task, variant)].append(float(raw[index]))

    rows: list[dict[str, Any]] = []
    for (split, task, variant), values in sorted(groups.items()):
        stats = summarize(np.asarray(values, dtype=np.float32))
        rows.append({"split": split, "task": task, "variant": variant, **stats})

    overall = summarize(raw)
    return rows, overall


def write_outputs(rows: list[dict[str, Any]], overall: dict[str, float], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "task",
        "variant",
        "num_windows",
        "delta_phi_raw_min",
        "delta_phi_raw_mean",
        "delta_phi_raw_max",
        "delta_phi_raw_std",
        "negative_rate",
        "stagnation_rate",
        "positive_rate",
    ]
    with (output_dir / "by_task_variant.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "variant_metrics.json").write_text(
        json.dumps({"overall": overall, "rows": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit RoboTwin rule-potential gains by task and perturbation variant.")
    parser.add_argument("--windows-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows, overall = audit_robotwin_variants(args.windows_dir)
    write_outputs(rows, overall, args.output_dir)
    print(json.dumps({"overall": overall, "num_groups": len(rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
