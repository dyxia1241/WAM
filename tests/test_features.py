import numpy as np
import pytest

from mvp0.features import read_feature_store, write_feature_store


def test_feature_store_round_trip(tmp_path):
    path = tmp_path / "features" / "ep0.npz"
    features = {
        "cam0": np.ones((5, 8), dtype=np.float16),
        "cam1": np.zeros((5, 8), dtype=np.float16),
    }

    write_feature_store(path, features)
    loaded = read_feature_store(path, expected_cameras=("cam0", "cam1"), expected_frames=5, expected_dim=8)

    assert set(loaded) == {"cam0", "cam1"}
    np.testing.assert_array_equal(loaded["cam0"], features["cam0"])


def test_feature_store_shape_mismatch_fails(tmp_path):
    path = tmp_path / "ep0.npz"
    write_feature_store(path, {"cam0": np.ones((5, 8), dtype=np.float16)})

    with pytest.raises(ValueError, match="expected 7"):
        read_feature_store(path, expected_frames=7)


def test_feature_store_missing_camera_fails(tmp_path):
    path = tmp_path / "ep0.npz"
    write_feature_store(path, {"cam0": np.ones((5, 8), dtype=np.float16)})

    with pytest.raises(ValueError, match="Missing camera"):
        read_feature_store(path, expected_cameras=("cam0", "cam1"))

