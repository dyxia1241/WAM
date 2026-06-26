import numpy as np
import pytest

from mvp0.counterfactual import ActionRange, make_simple_negative


def test_zero_reverse_and_shuffle_shapes():
    action = np.arange(24, dtype=np.float32).reshape(3, 8)
    rng = np.random.default_rng(0)

    assert make_simple_negative(action, "zero").shape == action.shape
    np.testing.assert_array_equal(make_simple_negative(action, "reverse"), action[::-1])
    assert make_simple_negative(action, "shuffle", rng=rng).shape == action.shape


def test_wrong_arm_swaps_halves():
    action = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)

    swapped = make_simple_negative(action, "wrong_arm")

    np.testing.assert_array_equal(swapped, np.array([[3, 4, 1, 2], [7, 8, 5, 6]], dtype=np.float32))


def test_wrong_arm_rejects_odd_action_dim():
    action = np.ones((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="even action dimension"):
        make_simple_negative(action, "wrong_arm")


def test_scaled_clips_to_action_range():
    action = np.array([[0.5, -0.5]], dtype=np.float32)
    action_range = ActionRange(
        low=np.array([-0.75, -0.75], dtype=np.float32),
        high=np.array([0.75, 0.75], dtype=np.float32),
    )

    scaled = make_simple_negative(action, "scaled", scale=2.0, action_range=action_range)

    np.testing.assert_array_equal(scaled, np.array([[0.75, -0.75]], dtype=np.float32))


def test_scaled_named_variants():
    action = np.array([[0.5, -0.5]], dtype=np.float32)

    small = make_simple_negative(action, "scaled_0.25")
    large = make_simple_negative(action, "scaled_1.75")

    np.testing.assert_array_equal(small, np.array([[0.125, -0.125]], dtype=np.float32))
    np.testing.assert_array_equal(large, np.array([[0.875, -0.875]], dtype=np.float32))


def test_shuffle_can_use_replacement_chunk():
    action = np.zeros((2, 4), dtype=np.float32)
    replacement = np.ones((2, 4), dtype=np.float32)

    shuffled = make_simple_negative(action, "shuffle", replacement_chunk=replacement)

    np.testing.assert_array_equal(shuffled, replacement)
