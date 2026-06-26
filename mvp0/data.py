from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


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


def collate_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    keys = samples[0].keys()
    return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in keys}
