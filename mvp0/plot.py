from __future__ import annotations

import argparse
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


def main() -> None:
    args = build_parser().parse_args()
    eval_dir = Path(args.eval)
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

    print(f"wrote {plot_dir / 'delta_phi_hist.png'}")


if __name__ == "__main__":
    main()
