import pytest

from ppwam.labels import (
    PrimitiveBoundary,
    compute_window_label,
    compute_window_label_from_potential,
    global_potential,
    primitive_phi,
    validate_boundaries,
)


def test_primitive_phi_clamps_to_unit_range():
    boundary = PrimitiveBoundary(stage="grasp", start=10, end=20)

    assert primitive_phi(10, boundary) == 0.0
    assert primitive_phi(15, boundary) == 0.5
    assert primitive_phi(20, boundary) == 1.0
    assert primitive_phi(25, boundary) == 1.0


def test_delta_phi_truncates_at_primitive_end():
    boundaries = [PrimitiveBoundary(stage="move", start=0, end=10)]

    label = compute_window_label(boundaries, t=8, horizon=8)

    assert label.primitive_time == pytest.approx(0.8)
    assert label.delta_phi == pytest.approx(0.2)
    assert label.cross_boundary is True
    assert label.stage == "move"


def test_delta_phi_inside_primitive():
    boundaries = [PrimitiveBoundary(stage="approach", start=0, end=20)]

    label = compute_window_label(boundaries, t=4, horizon=4)

    assert label.primitive_time == pytest.approx(0.2)
    assert label.delta_phi == pytest.approx(0.2)
    assert label.cross_boundary is False


def test_global_potential_tracks_primitive_chain():
    boundaries = [
        PrimitiveBoundary(stage="approach", start=0, end=10),
        PrimitiveBoundary(stage="grasp", start=11, end=20),
    ]

    assert global_potential(boundaries, 0) == pytest.approx(0.0)
    assert global_potential(boundaries, 10) == pytest.approx(0.5)
    assert global_potential(boundaries, 15) == pytest.approx((1.0 + 4.0 / 9.0) / 2.0)
    assert global_potential(boundaries, 20) == pytest.approx(1.0)


def test_window_label_exposes_absolute_potential_gain():
    boundaries = [
        PrimitiveBoundary(stage="approach", start=0, end=10),
        PrimitiveBoundary(stage="grasp", start=11, end=20),
    ]

    label = compute_window_label(boundaries, t=8, horizon=6)

    assert label.cross_boundary is True
    assert label.primitive_time == pytest.approx(0.8)
    assert label.delta_phi == pytest.approx(0.2)
    assert label.phi_t == pytest.approx(0.4)
    assert label.phi_future == pytest.approx((1.0 + 3.0 / 9.0) / 2.0)
    assert label.delta_phi_raw == pytest.approx(label.phi_future - label.phi_t)


def test_window_label_from_potential_allows_regression_gain():
    boundaries = [PrimitiveBoundary(stage="move", start=0, end=5)]
    potential = [0.5, 0.45, 0.40, 0.60, 0.70, 0.80]

    label = compute_window_label_from_potential(boundaries, potential, t=0, horizon=2)

    assert label.phi_t == pytest.approx(0.5)
    assert label.phi_future == pytest.approx(0.4)
    assert label.delta_phi_raw == pytest.approx(-0.1)
    assert label.delta_phi == pytest.approx(0.0)


def test_validate_boundaries_rejects_unknown_stage():
    with pytest.raises(ValueError, match="Unknown stage"):
        validate_boundaries([PrimitiveBoundary(stage="bad", start=0, end=1)])
