import pytest

from mvp0.labels import PrimitiveBoundary, compute_window_label, primitive_phi, validate_boundaries


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


def test_validate_boundaries_rejects_unknown_stage():
    with pytest.raises(ValueError, match="Unknown stage"):
        validate_boundaries([PrimitiveBoundary(stage="bad", start=0, end=1)])

