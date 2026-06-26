from __future__ import annotations

from pathlib import Path

import numpy as np


def write_feature_store(path: str | Path, features: dict[str, np.ndarray]) -> None:
    path = Path(path)
    if not features:
        raise ValueError("features must not be empty.")
    for camera, array in features.items():
        if array.ndim != 2:
            raise ValueError(f"Feature array for {camera} must have shape [T, D].")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **features)


def read_feature_store(
    path: str | Path,
    expected_cameras: tuple[str, ...] | None = None,
    expected_frames: int | None = None,
    expected_dim: int | None = None,
) -> dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path) as loaded:
        features = {camera: loaded[camera] for camera in loaded.files}

    if expected_cameras is not None:
        missing = set(expected_cameras) - set(features)
        if missing:
            raise ValueError(f"Missing camera features: {sorted(missing)}")

    for camera, array in features.items():
        if array.ndim != 2:
            raise ValueError(f"Feature array for {camera} must have shape [T, D].")
        if expected_frames is not None and array.shape[0] != expected_frames:
            raise ValueError(
                f"Feature array for {camera} has {array.shape[0]} frames; "
                f"expected {expected_frames}."
            )
        if expected_dim is not None and array.shape[1] != expected_dim:
            raise ValueError(
                f"Feature array for {camera} has dim {array.shape[1]}; expected {expected_dim}."
            )
    return features

