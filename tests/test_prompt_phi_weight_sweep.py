import pytest

from mvp0.prompt_phi_weight_sweep import aggregate_rows, parse_float_list, sample_std, weight_slug


def test_parse_float_list_and_weight_slug():
    assert parse_float_list("1,2.5,10") == [1.0, 2.5, 10.0]
    assert weight_slug(2.5) == "delta_w_2p5"


def test_aggregate_rows_groups_by_delta_weight():
    rows = [
        {"delta_weight": 1.0, "delta_phi_mae": 0.02, "delta_phi_rmse": 0.04},
        {"delta_weight": 1.0, "delta_phi_mae": 0.04, "delta_phi_rmse": 0.08},
        {"delta_weight": 5.0, "delta_phi_mae": 0.01, "delta_phi_rmse": 0.03},
    ]

    aggregates = aggregate_rows(rows)

    assert aggregates[0]["delta_weight"] == 1.0
    assert aggregates[0]["num_seeds"] == 2
    assert aggregates[0]["delta_phi_mae_mean"] == pytest.approx(0.03)
    assert aggregates[0]["delta_phi_mae_std"] == pytest.approx(0.0141421356)
    assert aggregates[0]["delta_phi_rmse_mean"] == pytest.approx(0.06)
    assert aggregates[1]["delta_weight"] == 5.0
    assert aggregates[1]["num_seeds"] == 1


def test_sample_std_returns_zero_for_single_value():
    assert sample_std([1.0]) == pytest.approx(0.0)
