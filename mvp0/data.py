from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from mvp0.features import read_feature_store
from mvp0.norm_stats import load_norm_stats, normalize_array


@dataclass(frozen=True)
class MockDatasetConfig:
    num_samples: int = 64
    history: int = 4
    horizon: int = 8
    cameras: int = 1
    feature_dim: int = 768
    proprio_dim: int = 14
    action_dim: int = 14
    num_stages: int = 5
    num_tasks: int = 2
    seed: int = 42
    split: str = "train"


class MockWindowDataset(Dataset):
    """Small deterministic dataset for CPU smoke tests and shape checks."""

    def __init__(self, config: MockDatasetConfig | None = None) -> None:
        self.config = config or MockDatasetConfig()
        rng = np.random.default_rng(self.config.seed)

        c = self.config
        self.obs_features = rng.normal(
            size=(c.num_samples, c.history, c.cameras, c.feature_dim)
        ).astype(np.float32)
        self.proprio = rng.normal(size=(c.num_samples, c.proprio_dim)).astype(np.float32)
        base = rng.normal(size=(c.num_samples, c.horizon, c.action_dim)).astype(np.float32)
        ramp = np.linspace(0.1, 1.0, c.horizon, dtype=np.float32)[None, :, None]
        self.action_chunk = np.clip(base * 0.2 + ramp, -1.0, 1.0).astype(np.float32)
        self.stage_id = rng.integers(0, c.num_stages, size=(c.num_samples,), dtype=np.int64)
        self.task_id = rng.integers(0, c.num_tasks, size=(c.num_samples,), dtype=np.int64)
        self.primitive_time = rng.uniform(0.0, 1.0, size=(c.num_samples,)).astype(np.float32)
        action_signal = self.action_chunk.mean(axis=(1, 2))
        stage_signal = (self.stage_id.astype(np.float32) + 1.0) / c.num_stages
        raw_delta = 0.05 + 0.35 * action_signal + 0.05 * self.primitive_time + 0.05 * stage_signal
        self.delta_phi = np.clip(raw_delta, 0.0, 1.0).astype(np.float32)
        self.split = np.array([c.split] * c.num_samples)

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "obs_features": torch.from_numpy(self.obs_features[index]),
            "proprio": torch.from_numpy(self.proprio[index]),
            "action_chunk": torch.from_numpy(self.action_chunk[index]),
            "stage_id": torch.tensor(self.stage_id[index], dtype=torch.long),
            "task_id": torch.tensor(self.task_id[index], dtype=torch.long),
            "primitive_time": torch.tensor(self.primitive_time[index], dtype=torch.float32),
            "delta_phi": torch.tensor(self.delta_phi[index], dtype=torch.float32),
        }


def make_mock_splits(config: MockDatasetConfig | None = None) -> dict[str, MockWindowDataset]:
    base = config or MockDatasetConfig()
    train = MockWindowDataset(base)
    val = MockWindowDataset(
        MockDatasetConfig(
            **{**base.__dict__, "num_samples": max(16, base.num_samples // 4), "seed": base.seed + 1, "split": "val"}
        )
    )
    test = MockWindowDataset(
        MockDatasetConfig(
            **{**base.__dict__, "num_samples": max(16, base.num_samples // 4), "seed": base.seed + 2, "split": "test"}
        )
    )
    return {"train": train, "val": val, "test": test}


class PreparedWindowDataset(Dataset):
    """Dataset backed by prepared windows, raw episode arrays, and feature stores."""

    def __init__(
        self,
        windows_dir: str | Path,
        episodes_dir: str | Path,
        features_dir: str | Path,
        split: str,
        feature_dim: int | None = None,
        norm_stats: str | Path | dict[str, Any] | None = None,
    ) -> None:
        self.windows_dir = Path(windows_dir)
        self.episodes_dir = Path(episodes_dir)
        self.features_dir = Path(features_dir)
        self.split = split
        self.feature_dim = feature_dim
        self.norm_stats = load_norm_stats(norm_stats) if norm_stats is not None else None

        self.windows = self._read_windows(self.windows_dir / "windows.jsonl")
        with np.load(self.windows_dir / "labels.npz") as labels:
            self.labels = {key: labels[key].copy() for key in labels.files}
        self.indices = [idx for idx, window in enumerate(self.windows) if window["split"] == split]
        if not self.indices:
            raise ValueError(f"No windows found for split={split}.")

        self._array_cache: dict[str, dict[str, np.ndarray]] = {}
        self._feature_cache: dict[str, dict[str, np.ndarray]] = {}

    @staticmethod
    def _read_windows(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(path)
        windows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    windows.append(json.loads(line))
        if not windows:
            raise ValueError(f"No windows found in {path}.")
        return windows

    def __len__(self) -> int:
        return len(self.indices)

    def _episode_arrays(self, episode_id: str) -> dict[str, np.ndarray]:
        if episode_id not in self._array_cache:
            path = self.episodes_dir / episode_id / "arrays.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path) as arrays:
                self._array_cache[episode_id] = {key: arrays[key].copy() for key in arrays.files}
            if "proprio" not in self._array_cache[episode_id] or "action" not in self._array_cache[episode_id]:
                raise ValueError(f"{path} must contain proprio and action.")
        return self._array_cache[episode_id]

    def _features(self, episode_id: str) -> dict[str, np.ndarray]:
        if episode_id not in self._feature_cache:
            self._feature_cache[episode_id] = read_feature_store(
                self.features_dir / f"{episode_id}.npz",
                expected_dim=self.feature_dim,
            )
        return self._feature_cache[episode_id]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        label_index = self.indices[index]
        window = self.windows[label_index]
        episode_id = str(window["episode_id"])
        history_indices = np.asarray(window["history_indices"], dtype=np.int64)
        future_indices = np.asarray(window["future_indices"], dtype=np.int64)
        t = int(window["t"])

        arrays = self._episode_arrays(episode_id)
        features = self._features(episode_id)
        camera_names = sorted(features)
        obs = np.stack([features[camera][history_indices] for camera in camera_names], axis=1)

        proprio = arrays["proprio"][t].astype(np.float32)
        action_chunk = arrays["action"][future_indices].astype(np.float32)
        if self.norm_stats is not None:
            proprio = normalize_array(proprio, self.norm_stats["proprio"])
            action_chunk = normalize_array(action_chunk, self.norm_stats["action"])

        return {
            "obs_features": torch.from_numpy(obs.astype(np.float32)),
            "proprio": torch.from_numpy(proprio),
            "action_chunk": torch.from_numpy(action_chunk),
            "stage_id": torch.tensor(int(self.labels["stage_id"][label_index]), dtype=torch.long),
            "task_id": torch.tensor(int(self.labels["task_id"][label_index]), dtype=torch.long),
            "primitive_time": torch.tensor(float(self.labels["primitive_time"][label_index]), dtype=torch.float32),
            "delta_phi": torch.tensor(float(self.labels["delta_phi"][label_index]), dtype=torch.float32),
        }


def collate_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    keys = samples[0].keys()
    return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in keys}
