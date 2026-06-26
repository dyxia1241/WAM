from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MVP-0 plotting entrypoint.")
    parser.add_argument("--eval", default="outputs/obs_action_stage_cf/eval", help="Evaluation output directory.")
    return parser


def read_predictions(path: Path) -> tuple[list[float], list[float]]:
    preds: list[float] = []
    targets: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            preds.append(float(record["pred_delta_phi"]))
            targets.append(float(record["target_delta_phi"]))
    return preds, targets


def read_margins(path: Path) -> dict[str, list[float]]:
    margins: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            margins.setdefault(row["negative_type"], []).append(float(row["margin"]))
    return margins


def read_stage_margins(path: Path) -> dict[str, list[float]]:
    margins: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            margins.setdefault(row["replacement_type"], []).append(float(row["margin"]))
    return margins


def main() -> None:
    args = build_parser().parse_args()
    run_plot(args.eval)


def run_plot(eval_dir: str | Path) -> list[Path]:
    eval_dir = Path(eval_dir)
    predictions_path = eval_dir / "predictions.jsonl"
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)

    preds, targets = read_predictions(predictions_path)
    plot_dir = eval_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.hist(preds, bins=20, alpha=0.7, label="pred_delta_phi")
    plt.hist(targets, bins=20, alpha=0.7, label="target_delta_phi")
    plt.xlabel("delta_phi")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "delta_phi_hist.png", dpi=150)
    plt.close()
    written = [plot_dir / "delta_phi_hist.png"]

    sensitivity_path = eval_dir / "action_sensitivity.csv"
    if sensitivity_path.exists():
        margins = read_margins(sensitivity_path)
        plt.figure(figsize=(7, 4))
        for negative_type, values in sorted(margins.items()):
            plt.hist(values, bins=20, alpha=0.45, label=negative_type)
        plt.axvline(0.0, color="black", linewidth=1)
        plt.xlabel("pred_delta_phi(correct) - pred_delta_phi(negative)")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "action_margin_hist.png", dpi=150)
        plt.close()
        written.append(plot_dir / "action_margin_hist.png")

    stage_path = eval_dir / "stage_sensitivity.csv"
    if stage_path.exists():
        margins = read_stage_margins(stage_path)
        plt.figure(figsize=(7, 4))
        for replacement_type, values in sorted(margins.items()):
            plt.hist(values, bins=20, alpha=0.45, label=replacement_type)
        plt.axvline(0.0, color="black", linewidth=1)
        plt.xlabel("pred_delta_phi(true_stage) - pred_delta_phi(wrong_stage)")
        plt.ylabel("count")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "stage_margin_hist.png", dpi=150)
        plt.close()
        written.append(plot_dir / "stage_margin_hist.png")

    print(f"wrote {plot_dir / 'delta_phi_hist.png'}")
    if sensitivity_path.exists():
        print(f"wrote {plot_dir / 'action_margin_hist.png'}")
    if stage_path.exists():
        print(f"wrote {plot_dir / 'stage_margin_hist.png'}")
    return written


if __name__ == "__main__":
    main()
