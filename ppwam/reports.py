from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_FIELDS = (
    "experiment",
    "delta_phi_mae",
    "delta_phi_rmse",
    "ranking_acc",
    "mean_margin",
    "true_vs_wrong_stage_margin",
    "wrong_stage_high_progress_rate",
    "git_commit",
    "checkpoint",
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected object in {path}.")
    return loaded


def collect_runs(outputs_dir: str | Path) -> list[dict[str, Any]]:
    outputs_dir = Path(outputs_dir)
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(outputs_dir.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        metrics = load_json(metrics_path)
        manifest_path = run_dir / "manifest.json"
        manifest = load_json(manifest_path) if manifest_path.exists() else {}
        experiment = str(manifest.get("experiment") or run_dir.name)
        row: dict[str, Any] = {"experiment": experiment}
        row.update(metrics)
        row["git_commit"] = manifest.get("git_commit")
        row["checkpoint"] = manifest.get("checkpoint")
        rows.append(row)
    return rows


def write_report(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(DEFAULT_FIELDS)
    extra_fields = sorted({key for row in rows for key in row if key not in fields})
    fields.extend(extra_fields)

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate PP-WAM run metrics.")
    parser.add_argument("--outputs", default="outputs")
    parser.add_argument("--output", default="outputs/report")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = collect_runs(args.outputs)
    write_report(rows, args.output)
    print(f"wrote report for {len(rows)} runs to {args.output}")


if __name__ == "__main__":
    main()
