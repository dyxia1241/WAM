from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ppwam.metrics import tie_aware_ranking


NEGATIVE_TYPES = ("zero", "shuffle", "wrong_arm", "scaled_0.25", "scaled_1.75", "reverse")
NUMERIC_KEYS = (
    "delta_phi_mae",
    "delta_phi_rmse",
    "all_negatives_tie_aware_ranking_acc",
    "all_negatives_mean_margin",
    "zero_ranking_acc",
    "zero_mean_margin",
    "shuffle_ranking_acc",
    "shuffle_mean_margin",
    "wrong_arm_ranking_acc",
    "wrong_arm_mean_margin",
    "scaled_0.25_ranking_acc",
    "scaled_0.25_mean_margin",
    "scaled_1.75_ranking_acc",
    "scaled_1.75_mean_margin",
    "reverse_ranking_acc",
    "reverse_mean_margin",
)

STAGE_EXPERIMENTS = (
    ("obs_stage", "stage_obs"),
    ("obs_action_stage", "stage_action"),
    ("obs_action_stage_cf_multi", "stage_action_cf"),
)
PROMPT_EXPERIMENTS = (
    ("obs_prompt", "prompt_obs"),
    ("obs_action_prompt", "prompt_action"),
    ("obs_action_prompt_cf_multi", "prompt_cf_w1"),
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return loaded


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def parse_seed(path: Path, manifest: dict[str, Any]) -> int:
    config_seed = manifest.get("config", {}).get("seed") if manifest else None
    if config_seed is not None:
        return int(config_seed)
    for part in path.parts:
        if part.startswith("seed_"):
            return int(part.removeprefix("seed_"))
    raise ValueError(f"Could not infer seed from {path}.")


def parse_weight_from_delta_dir(path: Path, manifest: dict[str, Any]) -> float | None:
    config_weight = manifest.get("config", {}).get("loss", {}).get("delta_weight") if manifest else None
    if config_weight is not None:
        return float(config_weight)
    for part in path.parts:
        if part.startswith("delta_w_"):
            return float(part.removeprefix("delta_w_").replace("p", ".").replace("m", "-"))
    return None


def action_sensitivity_summary(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}

    pos: list[float] = []
    neg: list[float] = []
    by_type: dict[str, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            negative_type = str(row["negative_type"])
            pos_value = float(row["pos_delta_phi"])
            neg_value = float(row["neg_delta_phi"])
            pos.append(pos_value)
            neg.append(neg_value)
            by_type[negative_type][0].append(pos_value)
            by_type[negative_type][1].append(neg_value)

    if not pos:
        return {}

    pos_arr = np.asarray(pos, dtype=np.float64)
    neg_arr = np.asarray(neg, dtype=np.float64)
    summary = {
        "all_negatives_tie_aware_ranking_acc": float(tie_aware_ranking(pos_arr, neg_arr)),
        "all_negatives_mean_margin": float(np.mean(pos_arr - neg_arr)),
    }
    for negative_type in NEGATIVE_TYPES:
        if negative_type not in by_type:
            continue
        type_pos = np.asarray(by_type[negative_type][0], dtype=np.float64)
        type_neg = np.asarray(by_type[negative_type][1], dtype=np.float64)
        summary[f"{negative_type}_ranking_acc"] = float(tie_aware_ranking(type_pos, type_neg))
        summary[f"{negative_type}_mean_margin"] = float(np.mean(type_pos - type_neg))
    return summary


def read_eval_record(
    eval_dir: Path,
    family: str,
    label: str,
    sort_key: int,
    delta_weight: float | None = None,
) -> dict[str, Any] | None:
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    metrics = read_json(metrics_path)
    manifest_path = eval_dir / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    experiment = str(manifest.get("experiment") or metrics.get("experiment") or eval_dir.parent.name)
    seed = parse_seed(eval_dir, manifest)
    resolved_weight = delta_weight
    if resolved_weight is None and family == "prompt_phi_weight_sweep":
        resolved_weight = parse_weight_from_delta_dir(eval_dir, manifest)

    record: dict[str, Any] = {
        "family": family,
        "label": label,
        "sort_key": sort_key,
        "experiment": experiment,
        "seed": seed,
        "delta_weight": resolved_weight,
        "eval_dir": str(eval_dir),
        "checkpoint": manifest.get("checkpoint"),
        "delta_phi_mae": float(metrics.get("prediction_delta_phi_mae", metrics["delta_phi_mae"])),
        "delta_phi_rmse": float(metrics.get("prediction_delta_phi_rmse", metrics["delta_phi_rmse"])),
    }
    record.update(action_sensitivity_summary(eval_dir / "action_sensitivity.csv"))
    return record


def collect_stage_records(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_dir in sorted(root.glob("seed_*")):
        for offset, (experiment, label) in enumerate(STAGE_EXPERIMENTS):
            record = read_eval_record(
                seed_dir / experiment / "eval",
                family="stage_first_experiment",
                label=label,
                sort_key=offset,
            )
            if record is not None:
                rows.append(record)
    return rows


def collect_prompt_records(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed_dir in sorted(root.glob("seed_*")):
        for offset, (experiment, label) in enumerate(PROMPT_EXPERIMENTS, start=10):
            record = read_eval_record(
                seed_dir / experiment / "eval_test",
                family="prompt_formal",
                label=label,
                sort_key=offset,
                delta_weight=1.0 if experiment == "obs_action_prompt_cf_multi" else None,
            )
            if record is not None:
                rows.append(record)
    return rows


def collect_sweep_records(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for delta_dir in sorted(root.glob("delta_w_*"), key=lambda path: parse_weight_from_delta_dir(path, {}) or 0.0):
        weight = parse_weight_from_delta_dir(delta_dir, {})
        if weight is None:
            continue
        for seed_dir in sorted(delta_dir.glob("seed_*")):
            record = read_eval_record(
                seed_dir / "obs_action_prompt_cf_multi" / "eval_test_action",
                family="prompt_phi_weight_sweep",
                label=f"prompt_cf_w{weight:g}",
                sort_key=100 + int(weight * 10),
                delta_weight=weight,
            )
            if record is not None:
                rows.append(record)
    return rows


def aggregate_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), str(row["label"]))].append(row)

    aggregates: list[dict[str, Any]] = []
    for (family, label), label_rows in grouped.items():
        first = min(label_rows, key=lambda row: int(row["seed"]))
        aggregate: dict[str, Any] = {
            "family": family,
            "label": label,
            "sort_key": int(first["sort_key"]),
            "experiment": first["experiment"],
            "delta_weight": first.get("delta_weight"),
            "num_seeds": len(label_rows),
        }
        for key in NUMERIC_KEYS:
            values = [float(row[key]) for row in label_rows if key in row and row[key] is not None]
            if values:
                aggregate[f"{key}_mean"] = float(np.mean(np.asarray(values, dtype=np.float64)))
                aggregate[f"{key}_std"] = sample_std(values)
        aggregates.append(aggregate)
    return sorted(aggregates, key=lambda row: int(row["sort_key"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "family",
        "label",
        "sort_key",
        "experiment",
        "delta_weight",
        "seed",
        "num_seeds",
        "delta_phi_mae",
        "delta_phi_rmse",
        "delta_phi_mae_mean",
        "delta_phi_mae_std",
        "delta_phi_rmse_mean",
        "delta_phi_rmse_std",
        "all_negatives_tie_aware_ranking_acc",
        "all_negatives_tie_aware_ranking_acc_mean",
        "all_negatives_tie_aware_ranking_acc_std",
        "all_negatives_mean_margin",
        "all_negatives_mean_margin_mean",
        "all_negatives_mean_margin_std",
    ]
    ordered = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)


def mean_std(row: dict[str, Any], key: str) -> str:
    mean_key = f"{key}_mean"
    std_key = f"{key}_std"
    if mean_key not in row:
        return "--"
    return f"{float(row[mean_key]):.4f}+/-{float(row.get(std_key, 0.0)):.4f}"


def plot_bar(
    aggregates: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    path: Path,
    rows_filter: str | None = None,
) -> None:
    rows = [row for row in aggregates if f"{metric}_mean" in row]
    if rows_filter == "sweep":
        rows = [row for row in rows if row["family"] == "prompt_phi_weight_sweep"]
    if not rows:
        return

    labels = [str(row["label"]) for row in rows]
    means = np.asarray([float(row[f"{metric}_mean"]) for row in rows], dtype=np.float64)
    stds = np.asarray([float(row.get(f"{metric}_std", 0.0)) for row in rows], dtype=np.float64)
    x = np.arange(len(rows))

    width = max(8.0, len(rows) * 0.85)
    fig, ax = plt.subplots(figsize=(width, 4.5))
    bars = ax.bar(x, means, yerr=stds, capsize=4, color="#4E79A7", edgecolor="#26394F", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for bar, mean in zip(bars, means, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{mean:.4f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def plot_sweep_per_negative(aggregates: list[dict[str, Any]], path: Path) -> None:
    rows = [row for row in aggregates if row["family"] == "prompt_phi_weight_sweep"]
    if not rows:
        return
    rows = sorted(rows, key=lambda row: float(row.get("delta_weight") or 0.0))
    x = np.asarray([float(row["delta_weight"]) for row in rows], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    for negative_type in NEGATIVE_TYPES:
        key = f"{negative_type}_ranking_acc_mean"
        if key not in rows[0]:
            continue
        y = np.asarray([float(row.get(key, np.nan)) for row in rows], dtype=np.float64)
        ax.plot(x, y, marker="o", linewidth=1.8, label=negative_type)
    ax.axhline(0.5, color="black", linewidth=1, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xticks(x, [f"{value:g}" for value in x])
    ax.set_xlabel("loss.delta_weight")
    ax.set_ylabel("Tie-aware ranking")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def write_figures(output_dir: Path, aggregates: list[dict[str, Any]]) -> None:
    figures = output_dir / "figures"
    comparison_rows = [
        row
        for row in aggregates
        if not (row["family"] == "prompt_phi_weight_sweep" and float(row.get("delta_weight") or 0.0) == 1.0)
    ]
    plot_bar(comparison_rows, "delta_phi_mae", "DeltaPhi MAE", figures / "comparison_delta_phi_mae.png")
    plot_bar(comparison_rows, "delta_phi_rmse", "DeltaPhi RMSE", figures / "comparison_delta_phi_rmse.png")
    plot_bar(
        comparison_rows,
        "all_negatives_tie_aware_ranking_acc",
        "All-negative tie-aware ranking",
        figures / "comparison_all_negative_ranking.png",
    )
    plot_bar(
        comparison_rows,
        "all_negatives_mean_margin",
        "All-negative mean margin",
        figures / "comparison_all_negative_margin.png",
    )
    plot_bar(
        aggregates,
        "all_negatives_tie_aware_ranking_acc",
        "All-negative tie-aware ranking",
        figures / "sweep_all_negative_ranking.png",
        rows_filter="sweep",
    )
    plot_bar(
        aggregates,
        "all_negatives_mean_margin",
        "All-negative mean margin",
        figures / "sweep_all_negative_margin.png",
        rows_filter="sweep",
    )
    plot_sweep_per_negative(aggregates, figures / "sweep_per_negative_ranking.png")


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| family | label | delta_weight | seeds | DeltaPhi MAE | DeltaPhi RMSE | all-neg ranking | all-neg margin | zero | shuffle | wrong_arm | scaled_1.75 | reverse |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        delta_weight = "--" if row.get("delta_weight") is None else f"{float(row['delta_weight']):g}"
        lines.append(
            "| {family} | `{label}` | {weight} | {seeds} | {mae} | {rmse} | {ranking} | {margin} | {zero} | {shuffle} | {wrong_arm} | {scaled} | {reverse} |".format(
                family=row["family"],
                label=row["label"],
                weight=delta_weight,
                seeds=int(row["num_seeds"]),
                mae=mean_std(row, "delta_phi_mae"),
                rmse=mean_std(row, "delta_phi_rmse"),
                ranking=mean_std(row, "all_negatives_tie_aware_ranking_acc"),
                margin=mean_std(row, "all_negatives_mean_margin"),
                zero=mean_std(row, "zero_ranking_acc"),
                shuffle=mean_std(row, "shuffle_ranking_acc"),
                wrong_arm=mean_std(row, "wrong_arm_ranking_acc"),
                scaled=mean_std(row, "scaled_1.75_ranking_acc"),
                reverse=mean_std(row, "reverse_ranking_acc"),
            )
        )
    return lines


def write_markdown_report(output_dir: Path, aggregates: list[dict[str, Any]]) -> None:
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    stage_rows = [row for row in aggregates if row["family"] == "stage_first_experiment"]
    prompt_rows = [row for row in aggregates if row["family"] == "prompt_formal"]
    sweep_rows = [row for row in aggregates if row["family"] == "prompt_phi_weight_sweep"]
    comparison_rows = stage_rows + prompt_rows + [row for row in sweep_rows if float(row["delta_weight"]) > 1.0]

    lines = [
        "# GM-100 Prompt Comprehensive Experiment Report",
        "",
        "This report merges the first stage-conditioned action experiment, the prompt-conditioned formal experiment, and the prompt CF DeltaPhi-loss weight sweep.",
        "",
        "## Metric Definitions",
        "",
        "- `DeltaPhi MAE/RMSE`: prediction error for primitive-local `DeltaPhi`.",
        "- `all-neg ranking`: tie-aware rate that positive action has higher predicted potential than generated negative actions.",
        "- `all-neg margin`: mean `pred_delta_phi(positive) - pred_delta_phi(negative)` over all evaluated negative types.",
        "- `+/-`: sample standard deviation across seeds.",
        "",
        "## Main Comparison",
        "",
    ]
    lines.extend(markdown_table(comparison_rows))
    lines.extend(
        [
            "",
            "## Phi Weight Sweep",
            "",
        ]
    )
    lines.extend(markdown_table(sweep_rows))
    lines.extend(
        [
            "",
            "## Main Findings",
            "",
            "- The original stage-conditioned CF model has stronger action ranking than non-CF action baselines, but worse DeltaPhi calibration than the prompt CF variants after increasing `loss.delta_weight`.",
            "- The prompt CF baseline at `loss.delta_weight=1` has high action sensitivity but poor DeltaPhi MAE/RMSE.",
            "- Raising `loss.delta_weight` improves DeltaPhi calibration monotonically in the tested range, while action ranking/margins drop toward random behavior.",
            "- `loss.delta_weight=10` and `20` are the best calibrated settings in this sweep; choose between them based on whether ranking sensitivity or DeltaPhi calibration is the primary objective.",
            "",
            "## Figures",
            "",
            "- `outputs/gm100_prompt_comprehensive_report/figures/comparison_delta_phi_mae.png`",
            "- `outputs/gm100_prompt_comprehensive_report/figures/comparison_delta_phi_rmse.png`",
            "- `outputs/gm100_prompt_comprehensive_report/figures/comparison_all_negative_ranking.png`",
            "- `outputs/gm100_prompt_comprehensive_report/figures/comparison_all_negative_margin.png`",
            "- `outputs/gm100_prompt_comprehensive_report/figures/sweep_all_negative_ranking.png`",
            "- `outputs/gm100_prompt_comprehensive_report/figures/sweep_all_negative_margin.png`",
            "- `outputs/gm100_prompt_comprehensive_report/figures/sweep_per_negative_ranking.png`",
        ]
    )
    (summary_dir / "experiment_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_report(stage_root: Path, prompt_root: Path, sweep_root: Path, output_dir: Path) -> list[dict[str, Any]]:
    rows = collect_stage_records(stage_root)
    rows.extend(collect_prompt_records(prompt_root))
    rows.extend(collect_sweep_records(sweep_root))
    aggregates = aggregate_records(rows)

    summary_dir = output_dir / "summary"
    write_csv(summary_dir / "metrics_by_seed.csv", rows)
    write_json(summary_dir / "metrics_by_seed.json", rows)
    write_csv(summary_dir / "aggregate_metrics.csv", aggregates)
    write_json(summary_dir / "aggregate_metrics.json", aggregates)
    write_figures(output_dir, aggregates)
    write_markdown_report(output_dir, aggregates)
    return aggregates


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build comprehensive GM-100 prompt experiment report.")
    parser.add_argument("--stage-root", default="outputs/gm100_action_fair_s42_44")
    parser.add_argument("--prompt-root", default="outputs/gm100_prompt_formal")
    parser.add_argument("--sweep-root", default="outputs/gm100_prompt_phi_weight_sweep")
    parser.add_argument("--output-dir", default="outputs/gm100_prompt_comprehensive_report")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    aggregates = build_report(
        stage_root=Path(args.stage_root),
        prompt_root=Path(args.prompt_root),
        sweep_root=Path(args.sweep_root),
        output_dir=Path(args.output_dir),
    )
    print(json.dumps(aggregates, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
