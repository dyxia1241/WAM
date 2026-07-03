import pytest

from ppwam.metrics import compute_metrics, summarize_by_type, tie_aware_ranking


def test_compute_metrics_with_ranking():
    metrics = compute_metrics(
        pred_delta_phi=[0.1, 0.4],
        target_delta_phi=[0.2, 0.2],
        pos_delta_phi=[0.8, 0.7],
        neg_delta_phi=[0.2, 0.9],
    )

    assert metrics["delta_phi_mae"] == pytest.approx(0.15)
    assert metrics["ranking_acc"] == pytest.approx(0.5)
    assert metrics["mean_margin"] == pytest.approx(0.2)


def test_tie_aware_ranking_counts_ties_as_half():
    assert tie_aware_ranking([0.5, 0.7, 0.1], [0.5, 0.3, 0.2]) == pytest.approx(0.5)


def test_summarize_by_type():
    summary = summarize_by_type(
        pos_delta_phi=[0.8, 0.1, 0.7],
        neg_delta_phi=[0.2, 0.2, 0.1],
        negative_types=["zero", "zero", "scaled"],
    )

    assert summary["zero_ranking_acc"] == pytest.approx(0.5)
    assert summary["scaled_ranking_acc"] == pytest.approx(1.0)
