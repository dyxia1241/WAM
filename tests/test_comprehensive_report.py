import csv

import pytest

from mvp0.comprehensive_report import action_sensitivity_summary, aggregate_records


def test_action_sensitivity_summary_computes_all_and_per_type_metrics(tmp_path):
    path = tmp_path / "action_sensitivity.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["negative_type", "pos_delta_phi", "neg_delta_phi", "margin", "is_correct"],
        )
        writer.writeheader()
        writer.writerows(
            [
                {"negative_type": "zero", "pos_delta_phi": 0.7, "neg_delta_phi": 0.2, "margin": 0.5, "is_correct": 1},
                {"negative_type": "zero", "pos_delta_phi": 0.4, "neg_delta_phi": 0.4, "margin": 0.0, "is_correct": 0},
                {"negative_type": "reverse", "pos_delta_phi": 0.1, "neg_delta_phi": 0.3, "margin": -0.2, "is_correct": 0},
            ]
        )

    summary = action_sensitivity_summary(path)

    assert summary["all_negatives_tie_aware_ranking_acc"] == pytest.approx(0.5)
    assert summary["all_negatives_mean_margin"] == pytest.approx(0.1)
    assert summary["zero_ranking_acc"] == pytest.approx(0.75)
    assert summary["reverse_ranking_acc"] == pytest.approx(0.0)


def test_aggregate_records_uses_sample_std():
    rows = [
        {
            "family": "prompt_phi_weight_sweep",
            "label": "prompt_cf_w2",
            "sort_key": 1,
            "experiment": "obs_action_prompt_cf_multi",
            "seed": 42,
            "delta_weight": 2.0,
            "delta_phi_mae": 0.01,
        },
        {
            "family": "prompt_phi_weight_sweep",
            "label": "prompt_cf_w2",
            "sort_key": 1,
            "experiment": "obs_action_prompt_cf_multi",
            "seed": 43,
            "delta_weight": 2.0,
            "delta_phi_mae": 0.03,
        },
    ]

    aggregates = aggregate_records(rows)

    assert aggregates[0]["delta_phi_mae_mean"] == pytest.approx(0.02)
    assert aggregates[0]["delta_phi_mae_std"] == pytest.approx(0.0141421356)
