from __future__ import annotations

import argparse
import csv
import json
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
    "zero_ranking_acc",
    "zero_mean_margin",
    "wrong_arm_ranking_acc",
    "wrong_arm_mean_margin",
    "scaled_0.25_ranking_acc",
    "scaled_0.25_mean_margin",
    "scaled_1.75_ranking_acc",
    "scaled_1.75_mean_margin",
    "reverse_ranking_acc",
    "reverse_mean_margin",
    "shuffle_ranking_acc",
    "shuffle_mean_margin",
)
COARSE_TYPES = ("zero", "wrong_arm", "scaled_0.25", "scaled_1.75")
TEMPORAL_TYPES = ("reverse", "shuffle")


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


def ensure_group_metrics(record: dict[str, Any]) -> None:
    for prefix, types in (("coarse_action_cf", COARSE_TYPES), ("temporal_diagnostic", TEMPORAL_TYPES)):
        ranking_key = f"{prefix}_ranking_acc"
        margin_key = f"{prefix}_mean_margin"
        if ranking_key in record and margin_key in record:
            continue
        rankings = [float(record[f"{kind}_ranking_acc"]) for kind in types if f"{kind}_ranking_acc" in record]
        margins = [float(record[f"{kind}_mean_margin"]) for kind in types if f"{kind}_mean_margin" in record]
        if rankings:
            record[ranking_key] = float(np.mean(np.asarray(rankings, dtype=np.float64)))
        if margins:
            record[margin_key] = float(np.mean(np.asarray(margins, dtype=np.float64)))


def read_run_record(run_dir: Path, label: str, family: str, seed: int) -> dict[str, Any] | None:
    metrics_path = run_dir / "mvp1_joint_flow" / "eval_test" / "metrics.json"
    if not metrics_path.exists():
        return None
    metrics = read_json(metrics_path)
    record: dict[str, Any] = {
        "family": family,
        "label": label,
        "seed": seed,
        "run_dir": str(run_dir / "mvp1_joint_flow"),
    }
    for key in NUMERIC_KEYS:
        if key in metrics:
            record[key] = float(metrics[key])
    ensure_group_metrics(record)
    return record


def collect_seeded_root(root: Path, label: str, family: str, seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        record = read_run_record(root / f"seed_{seed}", label=label, family=family, seed=seed)
        if record is not None:
            rows.append(record)
    return rows


def collect_named_root(root: Path, family: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for label_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        for seed_dir in sorted(label_dir.glob("seed_*")):
            try:
                seed = int(seed_dir.name.removeprefix("seed_"))
            except ValueError:
                continue
            record = read_run_record(seed_dir, label=label_dir.name, family=family, seed=seed)
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
        "all_negatives_tie_aware_ranking_acc",
        "all_negatives_mean_margin",
        "coarse_action_cf_ranking_acc",
        "coarse_action_cf_mean_margin",
        "temporal_diagnostic_ranking_acc",
        "temporal_diagnostic_mean_margin",
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


def plot_main(aggregates: list[dict[str, Any]], path: Path, labels: list[str]) -> None:
    rows = [row for row in aggregates if row["label"] in labels]
    rows = sorted(rows, key=lambda row: labels.index(str(row["label"])))
    metrics = [
        ("delta_phi_mae", "DeltaPhi MAE", (0.0, 0.03), None),
        ("delta_phi_rmse", "DeltaPhi RMSE", (0.0, 0.045), None),
        ("coarse_action_cf_ranking_acc", "Coarse CF ranking", (0.45, 1.0), 0.5),
        ("all_negatives_tie_aware_ranking_acc", "All-negative ranking", (0.45, 0.9), 0.5),
    ]
    colors = ["#4E79A7", "#59A14F", "#F28E2B", "#B07AA1", "#E15759", "#76B7B2"]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.8))
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
    fig.suptitle("GM-100 MVP1.5 Main Metrics", y=0.995)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_family(aggregates: list[dict[str, Any]], family: str, metric: str, title: str, path: Path) -> None:
    rows = [row for row in aggregates if row["family"] == family and f"{metric}_mean" in row]
    if not rows:
        return
    rows = sorted(rows, key=lambda row: str(row["label"]))
    values = np.asarray([float(row[f"{metric}_mean"]) for row in rows], dtype=np.float64)
    errors = np.asarray([float(row.get(f"{metric}_std", 0.0)) for row in rows], dtype=np.float64)
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(7.5, len(rows) * 1.0), 4.5))
    ax.bar(x, values, yerr=errors, capsize=4, color="#4E79A7", edgecolor="#26394F", linewidth=0.8)
    if metric.endswith("ranking_acc"):
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
        ax.set_ylim(0.45, 1.0)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels([str(row["label"]) for row in rows], rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_tradeoff(aggregates: list[dict[str, Any]], path: Path) -> None:
    rows = [
        row
        for row in aggregates
        if "delta_phi_mae_mean" in row and "coarse_action_cf_ranking_acc_mean" in row
    ]
    if not rows:
        return
    colors = {
        "main": "#4E79A7",
        "v3_formal": "#F28E2B",
        "ablation_seed42": "#E15759",
        "sweep_seed42": "#59A14F",
    }
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    for family in sorted({str(row["family"]) for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        x = [float(row["delta_phi_mae_mean"]) for row in family_rows]
        y = [float(row["coarse_action_cf_ranking_acc_mean"]) for row in family_rows]
        ax.scatter(
            x,
            y,
            s=70,
            label=family,
            color=colors.get(family, "#777777"),
            edgecolor="#222222",
            linewidth=0.7,
        )
        for row, x_value, y_value in zip(family_rows, x, y, strict=True):
            ax.annotate(
                str(row["label"]),
                xy=(x_value, y_value),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
            )
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("DeltaPhi MAE (lower is better)")
    ax.set_ylabel("Coarse action CF ranking (higher is better)")
    ax.set_title("GM-100 MVP1.5 Calibration vs Action Sensitivity")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def select_rows(aggregates: list[dict[str, Any]], family: str) -> list[dict[str, Any]]:
    return sorted([row for row in aggregates if row["family"] == family], key=lambda row: str(row["label"]))


def markdown_metric_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| label | seeds | MAE | RMSE | coarse ranking | coarse margin | all-neg ranking | all-neg margin | temporal ranking |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| `{label}` | {seeds} | {mae} | {rmse} | {coarse} | {coarse_margin} | {all_rank} | {all_margin} | {temporal} |".format(
                label=row["label"],
                seeds=int(row["num_seeds"]),
                mae=metric_pair(row, "delta_phi_mae"),
                rmse=metric_pair(row, "delta_phi_rmse"),
                coarse=metric_pair(row, "coarse_action_cf_ranking_acc"),
                coarse_margin=metric_pair(row, "coarse_action_cf_mean_margin"),
                all_rank=metric_pair(row, "all_negatives_tie_aware_ranking_acc"),
                all_margin=metric_pair(row, "all_negatives_mean_margin"),
                temporal=metric_pair(row, "temporal_diagnostic_ranking_acc"),
            )
        )
    return lines


def get_row(aggregates: list[dict[str, Any]], family: str, label: str) -> dict[str, Any] | None:
    for row in aggregates:
        if row["family"] == family and row["label"] == label:
            return row
    return None


def metric_value(row: dict[str, Any] | None, key: str) -> float | None:
    if row is None:
        return None
    value = row.get(f"{key}_mean")
    return float(value) if value is not None else None


def fmt_delta(value: float | None) -> str:
    if value is None:
        return "--"
    return f"{value:+.4f}"


def write_report(report_path: Path, aggregates: list[dict[str, Any]], figure_dir: Path) -> None:
    main_rows = [row for row in aggregates if row["family"] in {"main", "v3_formal"}]
    ablation_rows = select_rows(aggregates, "ablation_seed42")
    sweep_rows = select_rows(aggregates, "sweep_seed42")
    v2 = get_row(aggregates, "main", "mvp1_v2")
    v3 = get_row(aggregates, "v3_formal", "mvp1_v3_coarse")
    cf_1p0 = get_row(aggregates, "sweep_seed42", "cf_1p0")
    phi_w20 = get_row(aggregates, "sweep_seed42", "phi_w20")
    v2_coarse = metric_value(v2, "coarse_action_cf_ranking_acc")
    v3_coarse = metric_value(v3, "coarse_action_cf_ranking_acc")
    cf_1p0_coarse = metric_value(cf_1p0, "coarse_action_cf_ranking_acc")
    phi_w20_mae = metric_value(phi_w20, "delta_phi_mae")
    lines = [
        "# GM-100 MVP1.5 Experiment Report",
        "",
        "## Summary",
        "",
        "MVP1.5 separates label-faithful coarse action counterfactual metrics from temporal diagnostics and tests whether V2 should checkpoint on coarse action sensitivity.",
        "",
        "Main conclusions:",
        "",
        "- V3 coarse checkpoint selection reproduced the same three-seed aggregate as V2, so checkpoint selection alone is not a new improvement.",
        "- `cf_1p0` is the strongest seed-42 action-sensitivity pilot: coarse ranking reaches {cf_coarse} versus V2/V3 seed-42 `0.8050`, but MAE is worse than V2/V3.".format(
            cf_coarse="--" if cf_1p0_coarse is None else f"{cf_1p0_coarse:.4f}"
        ),
        "- `phi_w20` improves calibration on seed 42, with MAE {phi_mae}, but weakens coarse ranking, so it is not the best critic candidate.".format(
            phi_mae="--" if phi_w20_mae is None else f"{phi_w20_mae:.4f}"
        ),
        "- The temporal diagnostic group stays near chance for V1/V2/V3, which is expected under linearly interpolated primitive-time labels.",
        "",
        "Metric policy:",
        "",
        "- Primary action metric: `coarse action CF = zero + wrong_arm + scaled_0.25 + scaled_1.75`.",
        "- Diagnostic temporal metric: `temporal diagnostic = reverse + shuffle`.",
        "",
        "## Main Metrics",
        "",
        "| family | label | seeds | MAE | RMSE | coarse ranking | coarse margin | all-neg ranking | all-neg margin | temporal ranking |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(main_rows, key=lambda item: (str(item["family"]), str(item["label"]))):
        lines.append(
            "| {family} | `{label}` | {seeds} | {mae} | {rmse} | {coarse} | {coarse_margin} | {all_rank} | {all_margin} | {temporal} |".format(
                family=row["family"],
                label=row["label"],
                seeds=int(row["num_seeds"]),
                mae=metric_pair(row, "delta_phi_mae"),
                rmse=metric_pair(row, "delta_phi_rmse"),
                coarse=metric_pair(row, "coarse_action_cf_ranking_acc"),
                coarse_margin=metric_pair(row, "coarse_action_cf_mean_margin"),
                all_rank=metric_pair(row, "all_negatives_tie_aware_ranking_acc"),
                all_margin=metric_pair(row, "all_negatives_mean_margin"),
                temporal=metric_pair(row, "temporal_diagnostic_ranking_acc"),
            )
        )
    lines.extend(
        [
            "",
            "V3 vs V2 coarse-ranking delta: `{delta}`. This is effectively no change, so V3 should not replace V2 as a new method claim.".format(
                delta=fmt_delta(None if v2_coarse is None or v3_coarse is None else v3_coarse - v2_coarse)
            ),
            "",
            "## Seed-42 Component Ablation",
            "",
            "These runs are pilot ablations, not final statistics. `v2_full` and `v3_coarse` reuse the corresponding remote seed-42 formal outputs.",
            "",
            *markdown_metric_table(ablation_rows),
            "",
            "Ablation takeaway: stronger CF alone gives high ranking but destroys calibration; phi trajectory alone helps ranking but also hurts MAE; critic-flow auxiliary alone is insufficient. The full V2 recipe is the best balanced seed-42 model among these component tests.",
            "",
            "## Seed-42 Pilot Sweep",
            "",
            *markdown_metric_table(sweep_rows),
            "",
            "Sweep takeaway: `cf_1p0` is the only pilot that clearly improves action sensitivity, reaching the best coarse and all-negative ranking in this report. `phi_w20` and `critic_w2` improve calibration but reduce ranking. `steps_8` gives no meaningful gain over 4-step scoring while adding compute.",
            "",
            "## Recommendation",
            "",
            "Next run should be a three-seed formal sweep for `cf_1p0`, plus one calibration-preserving variant such as `cf_1p0 + phi_weight=20` or a checkpoint rule that jointly constrains MAE and coarse ranking. Do not promote V3 as a separate model; treat it as a selection-policy check that matched V2.",
            "",
            "## Figures",
            "",
            f"- `{figure_dir / 'mvp1_5_main_metrics.png'}`",
            f"- `{figure_dir / 'ablation_coarse_ranking.png'}`",
            f"- `{figure_dir / 'sweep_coarse_ranking.png'}`",
            f"- `{figure_dir / 'calibration_vs_coarse_ranking.png'}`",
            "",
            "## Files",
            "",
            f"- `{figure_dir / 'metrics_by_run.csv'}`",
            f"- `{figure_dir / 'aggregate_metrics.csv'}`",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate MVP1.5 joint-flow experiments.")
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--v1-root", default="outputs/gm100_mvp1_joint_flow")
    parser.add_argument("--v2-root", default="outputs/gm100_mvp1_joint_flow_v2")
    parser.add_argument("--v3-root", default="outputs/gm100_mvp1_joint_flow_v3_coarse")
    parser.add_argument("--ablation-root", default="outputs/gm100_mvp1_5_ablation")
    parser.add_argument("--sweep-root", default="outputs/gm100_mvp1_5_sweep")
    parser.add_argument("--docs-dir", default="docs/figures/gm100_mvp1_5")
    parser.add_argument("--report", default="docs/gm100_mvp1_5_report.md")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    seeds = parse_int_list(args.seeds)
    rows: list[dict[str, Any]] = []
    rows.extend(collect_seeded_root(Path(args.v1_root), "mvp1_v1", "main", seeds))
    rows.extend(collect_seeded_root(Path(args.v2_root), "mvp1_v2", "main", seeds))
    rows.extend(collect_seeded_root(Path(args.v3_root), "mvp1_v3_coarse", "v3_formal", seeds))
    rows.extend(collect_named_root(Path(args.ablation_root), "ablation_seed42"))
    rows.extend(collect_named_root(Path(args.sweep_root), "sweep_seed42"))
    if not rows:
        raise ValueError("No experiment rows found.")

    aggregates = aggregate_rows(rows)
    docs_dir = Path(args.docs_dir)
    write_csv(docs_dir / "metrics_by_run.csv", rows)
    write_json(docs_dir / "metrics_by_run.json", rows)
    write_csv(docs_dir / "aggregate_metrics.csv", aggregates)
    write_json(docs_dir / "aggregate_metrics.json", aggregates)
    plot_main(aggregates, docs_dir / "mvp1_5_main_metrics.png", ["mvp1_v1", "mvp1_v2", "mvp1_v3_coarse"])
    plot_family(
        aggregates,
        family="ablation_seed42",
        metric="coarse_action_cf_ranking_acc",
        title="MVP1.5 Seed-42 Ablation: Coarse CF Ranking",
        path=docs_dir / "ablation_coarse_ranking.png",
    )
    plot_family(
        aggregates,
        family="sweep_seed42",
        metric="coarse_action_cf_ranking_acc",
        title="MVP1.5 Seed-42 Sweep: Coarse CF Ranking",
        path=docs_dir / "sweep_coarse_ranking.png",
    )
    plot_tradeoff(aggregates, docs_dir / "calibration_vs_coarse_ranking.png")
    write_report(Path(args.report), aggregates, docs_dir)
    print(json.dumps({"num_rows": len(rows), "num_aggregates": len(aggregates)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
