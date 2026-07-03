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


NUMERIC_KEYS = (
    "delta_phi_mae",
    "delta_phi_rmse",
    "all_negatives_tie_aware_ranking_acc",
    "all_negatives_mean_margin",
    "coarse_action_cf_ranking_acc",
    "coarse_action_cf_mean_margin",
    "temporal_diagnostic_ranking_acc",
    "temporal_diagnostic_mean_margin",
    "all_negatives_top1_acc",
    "all_negatives_top1_margin",
    "coarse_action_cf_top1_acc",
    "coarse_action_cf_top1_margin",
    "temporal_diagnostic_top1_acc",
    "temporal_diagnostic_top1_margin",
    "zero_ranking_acc",
    "wrong_arm_ranking_acc",
    "scaled_0.25_ranking_acc",
    "scaled_1.75_ranking_acc",
    "reverse_ranking_acc",
    "shuffle_ranking_acc",
)
COARSE_TYPES = ("zero", "wrong_arm", "scaled_0.25", "scaled_1.75")
TEMPORAL_TYPES = ("reverse", "shuffle")
MAIN_LABELS = ("mvp1_v2", "cf_1p0", "cf1p0_phi_w20")


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


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


def group_metrics_from_per_type(record: dict[str, Any]) -> None:
    for prefix, types in (("coarse_action_cf", COARSE_TYPES), ("temporal_diagnostic", TEMPORAL_TYPES)):
        ranking_key = f"{prefix}_ranking_acc"
        margin_key = f"{prefix}_mean_margin"
        rankings = [float(record[f"{kind}_ranking_acc"]) for kind in types if f"{kind}_ranking_acc" in record]
        margins = [float(record[f"{kind}_mean_margin"]) for kind in types if f"{kind}_mean_margin" in record]
        if ranking_key not in record and rankings:
            record[ranking_key] = float(np.mean(np.asarray(rankings, dtype=np.float64)))
        if margin_key not in record and margins:
            record[margin_key] = float(np.mean(np.asarray(margins, dtype=np.float64)))


def top1_metrics(action_sensitivity_path: Path) -> dict[str, float]:
    if not action_sensitivity_path.exists():
        return {}

    by_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with action_sensitivity_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_index[int(row["index"])].append(
                {
                    "negative_type": row["negative_type"],
                    "pos_delta_phi": float(row["pos_delta_phi"]),
                    "neg_delta_phi": float(row["neg_delta_phi"]),
                }
            )

    metrics: dict[str, float] = {}
    for prefix, types in (
        ("all_negatives", None),
        ("coarse_action_cf", set(COARSE_TYPES)),
        ("temporal_diagnostic", set(TEMPORAL_TYPES)),
    ):
        acc: list[float] = []
        margins: list[float] = []
        for rows in by_index.values():
            filtered = [row for row in rows if types is None or row["negative_type"] in types]
            if not filtered:
                continue
            pos = float(filtered[0]["pos_delta_phi"])
            max_neg = max(float(row["neg_delta_phi"]) for row in filtered)
            acc.append(float(pos > max_neg))
            margins.append(pos - max_neg)
        if acc:
            metrics[f"{prefix}_top1_acc"] = float(np.mean(np.asarray(acc, dtype=np.float64)))
            metrics[f"{prefix}_top1_margin"] = float(np.mean(np.asarray(margins, dtype=np.float64)))
    return metrics


def read_run_record(run_dir: Path, label: str, family: str, seed: int) -> dict[str, Any] | None:
    eval_dir = run_dir / "mvp1_joint_flow" / "eval_test"
    metrics_path = eval_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = read_json(metrics_path)
    record: dict[str, Any] = {
        "family": family,
        "label": label,
        "seed": seed,
        "run_dir": str(run_dir / "mvp1_joint_flow"),
    }
    record.update(top1_metrics(eval_dir / "action_sensitivity.csv"))
    for key in NUMERIC_KEYS:
        if key in metrics:
            record[key] = float(metrics[key])
    group_metrics_from_per_type(record)
    return record


def collect_seeded_root(root: Path, label: str, family: str, seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        record = read_run_record(root / f"seed_{seed}", label=label, family=family, seed=seed)
        if record is not None:
            rows.append(record)
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["family"]), str(row["label"])), []).append(row)

    aggregates: list[dict[str, Any]] = []
    for (family, label), label_rows in grouped.items():
        aggregate: dict[str, Any] = {
            "family": family,
            "label": label,
            "num_seeds": len(label_rows),
        }
        for key in NUMERIC_KEYS:
            values = [float(row[key]) for row in label_rows if key in row]
            if values:
                aggregate[f"{key}_mean"] = float(np.mean(np.asarray(values, dtype=np.float64)))
                aggregate[f"{key}_std"] = sample_std(values)
        aggregates.append(aggregate)
    return sorted(aggregates, key=lambda row: (str(row["family"]), str(row["label"])))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    preferred = [
        "family",
        "label",
        "seed",
        "num_seeds",
        "delta_phi_mae",
        "delta_phi_rmse",
        "coarse_action_cf_ranking_acc",
        "all_negatives_tie_aware_ranking_acc",
        "coarse_action_cf_top1_acc",
        "all_negatives_top1_acc",
        "coarse_action_cf_mean_margin",
        "all_negatives_mean_margin",
    ]
    ordered = [field for field in preferred if field in fields] + [field for field in fields if field not in preferred]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, sort_keys=True)


def metric_pair(row: dict[str, Any], key: str) -> str:
    mean_key = f"{key}_mean"
    std_key = f"{key}_std"
    if mean_key not in row:
        return "--"
    return f"{float(row[mean_key]):.4f}+/-{float(row.get(std_key, 0.0)):.4f}"


def sorted_main_rows(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for row in aggregates if row["label"] in MAIN_LABELS]
    return sorted(rows, key=lambda row: MAIN_LABELS.index(str(row["label"])))


def plot_main(aggregates: list[dict[str, Any]], path: Path) -> None:
    rows = sorted_main_rows(aggregates)
    metrics = [
        ("delta_phi_mae", "DeltaPhi MAE", (0.0, 0.035), None),
        ("delta_phi_rmse", "DeltaPhi RMSE", (0.0, 0.05), None),
        ("coarse_action_cf_ranking_acc", "Coarse CF ranking", (0.45, 1.0), 0.5),
        ("all_negatives_tie_aware_ranking_acc", "All-negative ranking", (0.45, 0.9), 0.5),
        ("coarse_action_cf_top1_acc", "Coarse top-1 reranking", (0.0, 1.0), None),
        ("all_negatives_top1_acc", "All-negative top-1 reranking", (0.0, 1.0), None),
    ]
    colors = ["#4E79A7", "#59A14F", "#F28E2B"]
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.2))
    for ax, (metric, title, ylim, baseline) in zip(axes.reshape(-1), metrics, strict=True):
        values = np.asarray([float(row.get(f"{metric}_mean", np.nan)) for row in rows], dtype=np.float64)
        errors = np.asarray([float(row.get(f"{metric}_std", 0.0)) for row in rows], dtype=np.float64)
        x = np.arange(len(rows))
        ax.bar(x, values, yerr=errors, color=colors[: len(rows)], edgecolor="#333333", linewidth=0.8, capsize=4)
        if baseline is not None:
            ax.axhline(baseline, color="black", linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([str(row["label"]) for row in rows], rotation=25, ha="right")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("GM-100 MVP1.6 Formal Validation", y=0.995)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_tradeoff(aggregates: list[dict[str, Any]], path: Path) -> None:
    rows = [row for row in aggregates if "delta_phi_mae_mean" in row and "coarse_action_cf_ranking_acc_mean" in row]
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    colors = {"baseline": "#4E79A7", "formal_validation": "#59A14F"}
    for family in sorted({str(row["family"]) for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        ax.scatter(
            [float(row["delta_phi_mae_mean"]) for row in family_rows],
            [float(row["coarse_action_cf_ranking_acc_mean"]) for row in family_rows],
            s=75,
            label=family,
            color=colors.get(family, "#777777"),
            edgecolor="#222222",
            linewidth=0.7,
        )
        for row in family_rows:
            ax.annotate(
                str(row["label"]),
                xy=(float(row["delta_phi_mae_mean"]), float(row["coarse_action_cf_ranking_acc_mean"])),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=8,
            )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("DeltaPhi MAE (lower is better)")
    ax.set_ylabel("Coarse action CF ranking (higher is better)")
    ax.set_title("MVP1.6 Calibration vs Action Sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def markdown_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| family | label | seeds | MAE | RMSE | coarse ranking | all-neg ranking | coarse top-1 | all-neg top-1 | coarse margin |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {family} | `{label}` | {seeds} | {mae} | {rmse} | {coarse} | {all_rank} | {coarse_top1} | {all_top1} | {coarse_margin} |".format(
                family=row["family"],
                label=row["label"],
                seeds=int(row["num_seeds"]),
                mae=metric_pair(row, "delta_phi_mae"),
                rmse=metric_pair(row, "delta_phi_rmse"),
                coarse=metric_pair(row, "coarse_action_cf_ranking_acc"),
                all_rank=metric_pair(row, "all_negatives_tie_aware_ranking_acc"),
                coarse_top1=metric_pair(row, "coarse_action_cf_top1_acc"),
                all_top1=metric_pair(row, "all_negatives_top1_acc"),
                coarse_margin=metric_pair(row, "coarse_action_cf_mean_margin"),
            )
        )
    return lines


def write_report(report_path: Path, aggregates: list[dict[str, Any]], figure_dir: Path) -> None:
    rows = sorted_main_rows(aggregates)
    cf_row = next((row for row in rows if row["label"] == "cf_1p0"), None)
    phi_row = next((row for row in rows if row["label"] == "cf1p0_phi_w20"), None)
    v2_row = next((row for row in rows if row["label"] == "mvp1_v2"), None)

    def mean(row: dict[str, Any] | None, key: str) -> float | None:
        if row is None:
            return None
        value = row.get(f"{key}_mean")
        return float(value) if value is not None else None

    cf_delta = None
    if cf_row is not None and v2_row is not None:
        cf_delta = mean(cf_row, "coarse_action_cf_ranking_acc") - mean(v2_row, "coarse_action_cf_ranking_acc")
    phi_mae_delta = None
    if phi_row is not None and cf_row is not None:
        phi_mae_delta = mean(phi_row, "delta_phi_mae") - mean(cf_row, "delta_phi_mae")

    lines = [
        "# GM-100 MVP1.6 Formal Validation Report",
        "",
        "## Summary",
        "",
        "This validation runs the strongest MVP1.5 action-sensitivity pilot across three seeds and tests a calibration-preserving variant.",
        "",
        "Candidate configs:",
        "",
        "- `cf_1p0`: V2/V3 recipe with `counterfactual_weight=1.0`.",
        "- `cf1p0_phi_w20`: same as `cf_1p0`, but `phi_weight=20.0`.",
        "",
        "## Main Metrics",
        "",
        *markdown_table(rows),
        "",
        "## Interpretation",
        "",
        "- `cf_1p0` coarse-ranking delta over V2: `{}`.".format("--" if cf_delta is None else f"{cf_delta:+.4f}"),
        "- `cf1p0_phi_w20` MAE delta versus `cf_1p0`: `{}`.".format(
            "--" if phi_mae_delta is None else f"{phi_mae_delta:+.4f}"
        ),
        "- Top-1 reranking measures whether the positive action is scored above every generated candidate negative for the same state.",
        "",
        "## Figures",
        "",
        f"- `{figure_dir / 'mvp1_6_main_metrics.png'}`",
        f"- `{figure_dir / 'calibration_vs_coarse_ranking.png'}`",
        "",
        "## Files",
        "",
        f"- `{figure_dir / 'metrics_by_run.csv'}`",
        f"- `{figure_dir / 'aggregate_metrics.csv'}`",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate MVP1.6 joint-flow validation experiments.")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--v2-root", default="outputs/gm100_mvp1_joint_flow_v2")
    parser.add_argument("--cf1p0-root", default="outputs/gm100_mvp1_6_cf1p0")
    parser.add_argument("--cf1p0-phi-w20-root", default="outputs/gm100_mvp1_6_cf1p0_phi_w20")
    parser.add_argument("--docs-dir", default="docs/figures/gm100_mvp1_6_validation")
    parser.add_argument("--report", default="docs/gm100_mvp1_6_validation_report.md")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = parse_int_list(args.seeds)
    rows: list[dict[str, Any]] = []
    rows.extend(collect_seeded_root(Path(args.v2_root), "mvp1_v2", "baseline", seeds))
    rows.extend(collect_seeded_root(Path(args.cf1p0_root), "cf_1p0", "formal_validation", seeds))
    rows.extend(collect_seeded_root(Path(args.cf1p0_phi_w20_root), "cf1p0_phi_w20", "formal_validation", seeds))
    if not rows:
        raise ValueError("No experiment rows found.")

    aggregates = aggregate_rows(rows)
    docs_dir = Path(args.docs_dir)
    write_csv(docs_dir / "metrics_by_run.csv", rows)
    write_json(docs_dir / "metrics_by_run.json", rows)
    write_csv(docs_dir / "aggregate_metrics.csv", aggregates)
    write_json(docs_dir / "aggregate_metrics.json", aggregates)
    plot_main(aggregates, docs_dir / "mvp1_6_main_metrics.png")
    plot_tradeoff(aggregates, docs_dir / "calibration_vs_coarse_ranking.png")
    write_report(Path(args.report), aggregates, docs_dir)
    print(json.dumps({"num_rows": len(rows), "num_aggregates": len(aggregates)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
