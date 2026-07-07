from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ppwam.features import read_feature_store
from ppwam.norm_stats import load_norm_stats, normalize_array
from ppwam.prompts import load_prompt_feature_store


@dataclass(frozen=True)
class MockDatasetConfig:
    num_samples: int = 64
    history: int = 4
    horizon: int = 8
    cameras: int = 1
    feature_dim: int = 768
    proprio_dim: int = 14
    action_dim: int = 14
    prompt_dim: int = 512
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
        task_prompt_features = rng.normal(size=(c.num_tasks, c.prompt_dim)).astype(np.float32)
        self.prompt_features = task_prompt_features[self.task_id]
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
            "prompt_features": torch.from_numpy(self.prompt_features[index]),
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
        prompt_features: str | Path | None = None,
        prompt_feature_dim: int | None = None,
        canonical_proprio_dim: int | None = None,
        canonical_action_dim: int | None = None,
    ) -> None:
        self.windows_dir = Path(windows_dir)
        self.episodes_dir = Path(episodes_dir)
        self.features_dir = Path(features_dir)
        self.split = split
        self.feature_dim = feature_dim
        self.norm_stats = load_norm_stats(norm_stats) if norm_stats is not None else None
        self.index = self._read_index()
        self.source_specs = self._read_source_specs()
        self.source_norm_stats = self._load_source_norm_stats()
        index_params = self.index.get("params", {}) if isinstance(self.index.get("params", {}), dict) else {}
        self.canonical_proprio_dim = (
            int(canonical_proprio_dim)
            if canonical_proprio_dim is not None
            else (int(index_params["canonical_proprio_dim"]) if index_params.get("canonical_proprio_dim") is not None else None)
        )
        self.canonical_action_dim = (
            int(canonical_action_dim)
            if canonical_action_dim is not None
            else (int(index_params["canonical_action_dim"]) if index_params.get("canonical_action_dim") is not None else None)
        )
        self.prompt_feature_map = (
            load_prompt_feature_store(prompt_features, expected_dim=prompt_feature_dim)
            if prompt_features is not None
            else None
        )

        self.windows = self._read_windows(self.windows_dir / "windows.jsonl")
        with np.load(self.windows_dir / "labels.npz") as labels:
            self.labels = {key: labels[key].copy() for key in labels.files}
        self.indices = [idx for idx, window in enumerate(self.windows) if window["split"] == split]
        if not self.indices:
            raise ValueError(f"No windows found for split={split}.")

        self._array_cache: dict[str, dict[str, np.ndarray]] = {}
        self._feature_cache: dict[str, dict[str, np.ndarray]] = {}

    def _read_index(self) -> dict[str, Any]:
        path = self.windows_dir / "index.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected JSON object in {path}.")
        return loaded

    def _read_source_specs(self) -> dict[str, dict[str, Any]]:
        raw = self.index.get("sources", {})
        if raw is None:
            return {}
        if not isinstance(raw, dict):
            raise ValueError("index.json sources must be an object.")
        out: dict[str, dict[str, Any]] = {}
        for source, spec in raw.items():
            if not isinstance(spec, dict):
                raise ValueError(f"index.json source spec for {source!r} must be an object.")
            out[str(source)] = dict(spec)
        return out

    @staticmethod
    def _path_from_spec(value: Any) -> Path | None:
        if value in (None, ""):
            return None
        return Path(str(value))

    def _load_source_norm_stats(self) -> dict[str, dict[str, Any]]:
        stats: dict[str, dict[str, Any]] = {}
        for source, spec in self.source_specs.items():
            path = self._path_from_spec(spec.get("norm_stats"))
            if path is not None:
                stats[source] = load_norm_stats(path)
        return stats

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

    def _source_name(self, window: dict[str, Any]) -> str:
        return str(window.get("source", ""))

    def _episodes_dir(self, source: str) -> Path:
        spec = self.source_specs.get(source, {})
        path = self._path_from_spec(spec.get("episodes_dir"))
        return path if path is not None else self.episodes_dir

    def _features_dir(self, source: str) -> Path:
        spec = self.source_specs.get(source, {})
        path = self._path_from_spec(spec.get("features_dir"))
        return path if path is not None else self.features_dir

    def _norm_stats(self, source: str) -> dict[str, Any] | None:
        return self.source_norm_stats.get(source, self.norm_stats)

    @staticmethod
    def _pad_last_dim(values: np.ndarray, target_dim: int | None, name: str) -> np.ndarray:
        if target_dim is None:
            return values
        current = int(values.shape[-1])
        if current == int(target_dim):
            return values
        if current > int(target_dim):
            raise ValueError(f"{name} dim {current} exceeds canonical dim {target_dim}.")
        pad_width = [(0, 0)] * values.ndim
        pad_width[-1] = (0, int(target_dim) - current)
        return np.pad(values, pad_width, mode="constant").astype(np.float32)

    def _episode_arrays(self, episode_id: str, source: str = "") -> dict[str, np.ndarray]:
        cache_key = f"{source}\0{episode_id}"
        if cache_key not in self._array_cache:
            path = self._episodes_dir(source) / episode_id / "arrays.npz"
            if not path.exists():
                raise FileNotFoundError(path)
            with np.load(path) as arrays:
                self._array_cache[cache_key] = {key: arrays[key].copy() for key in arrays.files}
            if "proprio" not in self._array_cache[cache_key] or "action" not in self._array_cache[cache_key]:
                raise ValueError(f"{path} must contain proprio and action.")
        return self._array_cache[cache_key]

    def _features(self, episode_id: str, source: str = "") -> dict[str, np.ndarray]:
        cache_key = f"{source}\0{episode_id}"
        if cache_key not in self._feature_cache:
            self._feature_cache[cache_key] = read_feature_store(
                self._features_dir(source) / f"{episode_id}.npz",
                expected_dim=self.feature_dim,
            )
        return self._feature_cache[cache_key]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        label_index = self.indices[index]
        window = self.windows[label_index]
        source = self._source_name(window)
        episode_id = str(window["episode_id"])
        history_indices = np.asarray(window["history_indices"], dtype=np.int64)
        future_indices = np.asarray(window["future_indices"], dtype=np.int64)
        t = int(window["t"])

        arrays = self._episode_arrays(episode_id, source=source)
        features = self._features(episode_id, source=source)
        camera_names = sorted(features)
        obs = np.stack([features[camera][history_indices] for camera in camera_names], axis=1)

        proprio = arrays["proprio"][t].astype(np.float32)
        action_chunk = arrays["action"][future_indices].astype(np.float32)
        norm_stats = self._norm_stats(source)
        if norm_stats is not None:
            proprio = normalize_array(proprio, norm_stats["proprio"])
            action_chunk = normalize_array(action_chunk, norm_stats["action"])
        proprio = self._pad_last_dim(proprio, self.canonical_proprio_dim, "proprio")
        action_chunk = self._pad_last_dim(action_chunk, self.canonical_action_dim, "action")
        if "source_id" in self.labels:
            source_id = int(self.labels["source_id"][label_index])
        else:
            source_id = int(window.get("source_id", -1))

        sample = {
            "obs_features": torch.from_numpy(obs.astype(np.float32)),
            "proprio": torch.from_numpy(proprio),
            "action_chunk": torch.from_numpy(action_chunk),
            "stage_id": torch.tensor(int(self.labels["stage_id"][label_index]), dtype=torch.long),
            "task_id": torch.tensor(int(self.labels["task_id"][label_index]), dtype=torch.long),
            "source_id": torch.tensor(source_id, dtype=torch.long),
            "primitive_time": torch.tensor(float(self.labels["primitive_time"][label_index]), dtype=torch.float32),
            "delta_phi": torch.tensor(float(self.labels["delta_phi"][label_index]), dtype=torch.float32),
        }
        if self.prompt_feature_map is not None:
            raw_task_id = str(window["task_id"])
            if raw_task_id not in self.prompt_feature_map:
                raise KeyError(f"Prompt features missing task_id={raw_task_id}")
            sample["prompt_features"] = torch.from_numpy(self.prompt_feature_map[raw_task_id])
        return sample


def collate_batch(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    keys = samples[0].keys()
    return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in keys}
