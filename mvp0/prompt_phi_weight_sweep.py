from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mvp0.config import apply_overrides, load_config
from mvp0.eval import write_predictions
from mvp0.manifest import write_manifest
from mvp0.train import build_model, make_loaders, train


EXPERIMENT = "obs_action_prompt_cf_multi"
DEFAULT_WEIGHTS = "1,2,5,10,20"
DEFAULT_SEEDS = "42,43,44"


def parse_float_list(value: str) -> list[float]:
    parsed = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one numeric value.")
    return parsed


def parse_int_list(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise ValueError("Expected at least one integer value.")
    return parsed


def weight_label(weight: float) -> str:
    return f"{weight:g}"


def weight_slug(weight: float) -> str:
    return "delta_w_" + weight_label(weight).replace(".", "p").replace("-", "m")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return loaded


def sample_std(values: list[float] | np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[0] < 2:
        return 0.0
    return float(np.std(array, ddof=1))


@torch.no_grad()
def evaluate_predict_phi_only(
    checkpoint_path: str | Path,
    split: str = "test",
    output_dir: str | Path | None = None,
) -> dict[str, float]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    experiment = checkpoint["experiment"]
    if experiment != EXPERIMENT:
        raise ValueError(f"Expected {EXPERIMENT}, got {experiment}.")

    device = torch.device(config.get("device", "cpu"))
    loaders = make_loaders(config)
    if split not in loaders:
        raise ValueError(f"Unknown split: {split}")

    model = build_model(config, experiment).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    eval_dir = Path(output_dir) if output_dir else checkpoint_path.parent / f"eval_{split}_predict_phi"
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics = write_predictions(
        model,
        loaders[split],
        experiment,
        device,
        eval_dir / "predictions.jsonl",
    )
    metrics["split"] = split
    metrics["delta_weight"] = float(config.get("loss", {}).get("delta_weight", 1.0))
    with (eval_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    write_manifest(
        eval_dir / "manifest.json",
        kind="eval_predict_phi",
        config=config,
        metrics=metrics,
        experiment=experiment,
        checkpoint=str(checkpoint_path),
        split=split,
        repo_root=Path(__file__).resolve().parents[1],
    )
    return metrics


def make_run_config(base_config: dict[str, Any], output_dir: Path, seed: int, delta_weight: float) -> dict[str, Any]:
    run_config = copy.deepcopy(base_config)
    run_config["experiment"] = EXPERIMENT
    run_config["seed"] = seed
    run_config["output_dir"] = str(output_dir)
    run_config.setdefault("loss", {})["delta_weight"] = float(delta_weight)
    return run_config


def run_one(
    base_config: dict[str, Any],
    root: Path,
    seed: int,
    delta_weight: float,
    split: str,
    force: bool = False,
) -> dict[str, Any]:
    seed_root = root / weight_slug(delta_weight) / f"seed_{seed}"
    run_dir = seed_root / EXPERIMENT
    eval_dir = run_dir / f"eval_{split}_predict_phi"
    eval_metrics_path = eval_dir / "metrics.json"
    checkpoint_path = run_dir / "best.pt"

    if force or not eval_metrics_path.exists():
        if force or not checkpoint_path.exists():
            run_config = make_run_config(base_config, seed_root, seed, delta_weight)
            train(run_config)
        evaluate_predict_phi_only(checkpoint_path, split=split, output_dir=eval_dir)

    metrics = load_json(eval_metrics_path)
    return {
        "delta_weight": float(delta_weight),
        "delta_weight_label": weight_label(delta_weight),
        "seed": seed,
        "experiment": EXPERIMENT,
        "checkpoint": str(checkpoint_path),
        "eval_dir": str(eval_dir),
        "delta_phi_mae": float(metrics["delta_phi_mae"]),
        "delta_phi_rmse": float(metrics["delta_phi_rmse"]),
    }


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    weights = sorted({float(row["delta_weight"]) for row in rows})
    for weight in weights:
        weight_rows = [row for row in rows if float(row["delta_weight"]) == weight]
        mae = np.asarray([float(row["delta_phi_mae"]) for row in weight_rows], dtype=np.float64)
        rmse = np.asarray([float(row["delta_phi_rmse"]) for row in weight_rows], dtype=np.float64)
        aggregates.append(
            {
                "delta_weight": weight,
                "delta_weight_label": weight_label(weight),
                "num_seeds": len(weight_rows),
                "delta_phi_mae_mean": float(np.mean(mae)),
                "delta_phi_mae_std": sample_std(mae),
                "delta_phi_rmse_mean": float(np.mean(rmse)),
                "delta_phi_rmse_std": sample_std(rmse),
            }
        )
    return aggregates


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write.")
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)


def plot_metric(aggregates: list[dict[str, Any]], metric: str, ylabel: str, path: Path) -> None:
    ordered = sorted(aggregates, key=lambda row: float(row["delta_weight"]))
    labels = [str(row["delta_weight_label"]) for row in ordered]
    means = np.asarray([float(row[f"{metric}_mean"]) for row in ordered], dtype=np.float64)
    stds = np.asarray([float(row[f"{metric}_std"]) for row in ordered], dtype=np.float64)
    x = np.arange(len(ordered))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="#4E79A7", edgecolor="#26394F", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_xlabel("loss.delta_weight")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{EXPERIMENT}: {ylabel}")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, mean in zip(bars, means, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{mean:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def write_markdown_report(root: Path, aggregates: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    summary_dir = root / "summary"
    lines = [
        "# GM-100 Prompt Phi Weight Sweep",
        "",
        f"- Experiment: `{EXPERIMENT}`",
        "- Evaluation: prediction-only test DeltaPhi metrics",
        "- Changed parameter: `loss.delta_weight` in total loss",
        "- Fixed parameter: `loss.counterfactual_weight=0.1` unless overridden in the config",
        "",
        "| loss.delta_weight | seeds | DeltaPhi MAE | DeltaPhi RMSE |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(aggregates, key=lambda item: float(item["delta_weight"])):
        lines.append(
            "| {weight} | {seeds} | {mae:.4f}+/-{mae_std:.4f} | {rmse:.4f}+/-{rmse_std:.4f} |".format(
                weight=row["delta_weight_label"],
                seeds=int(row["num_seeds"]),
                mae=float(row["delta_phi_mae_mean"]),
                mae_std=float(row["delta_phi_mae_std"]),
                rmse=float(row["delta_phi_rmse_mean"]),
                rmse_std=float(row["delta_phi_rmse_std"]),
            )
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            f"- `{root / 'figures' / 'test_delta_phi_mae.png'}`",
            f"- `{root / 'figures' / 'test_delta_phi_rmse.png'}`",
            "",
            "## Per-Seed Files",
            "",
        ]
    )
    for row in sorted(rows, key=lambda item: (float(item["delta_weight"]), int(item["seed"]))):
        lines.append(f"- weight {row['delta_weight_label']}, seed {row['seed']}: `{row['eval_dir']}`")
    (summary_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary(root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates = aggregate_rows(rows)
    summary_dir = root / "summary"
    figures_dir = root / "figures"
    write_csv(summary_dir / "eval_test_predict_phi_by_seed.csv", rows)
    write_json(summary_dir / "eval_test_predict_phi_by_seed.json", rows)
    write_csv(summary_dir / "aggregate_eval_test_predict_phi.csv", aggregates)
    write_json(summary_dir / "aggregate_eval_test_predict_phi.json", aggregates)
    plot_metric(aggregates, "delta_phi_mae", "DeltaPhi MAE", figures_dir / "test_delta_phi_mae.png")
    plot_metric(aggregates, "delta_phi_rmse", "DeltaPhi RMSE", figures_dir / "test_delta_phi_rmse.png")
    write_markdown_report(root, aggregates, rows)
    return aggregates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep DeltaPhi loss weight for GM-100 prompt CF critic.")
    parser.add_argument("--config", default="mvp0/configs/gm100_prompt_formal.yaml")
    parser.add_argument("--output-dir", default="outputs/gm100_prompt_phi_weight_sweep")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Comma-separated loss.delta_weight values.")
    parser.add_argument("--seeds", default=DEFAULT_SEEDS, help="Comma-separated random seeds.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--force", action="store_true", help="Rerun even if prediction metrics already exist.")
    parser.add_argument("overrides", nargs="*", help="Optional config key=value overrides.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_config = apply_overrides(load_config(args.config), args.overrides)
    root = Path(args.output_dir)
    rows: list[dict[str, Any]] = []
    for delta_weight in parse_float_list(args.weights):
        for seed in parse_int_list(args.seeds):
            rows.append(
                run_one(
                    base_config=base_config,
                    root=root,
                    seed=seed,
                    delta_weight=delta_weight,
                    split=args.split,
                    force=args.force,
                )
            )
    aggregates = write_summary(root, rows)
    print(json.dumps(aggregates, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
