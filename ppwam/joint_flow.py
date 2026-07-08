from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ppwam.config import apply_overrides, load_config
from ppwam.counterfactual import make_negative_action_tensor
from ppwam.data import MockDatasetConfig, MockWindowDataset, PreparedWindowDataset, collate_batch
from ppwam.manifest import write_manifest
from ppwam.metrics import compute_metrics, summarize_by_type, tie_aware_ranking
from ppwam.norm_stats import normalize_array
from ppwam.train import batch_to_device, expand_negative_types, parse_negative_types, score_for_checkpoint, set_seed


EXPERIMENT = "mvp1_joint_flow"
COARSE_NEGATIVE_TYPES = ("zero", "wrong_arm", "scaled_0.25", "scaled_1.75")
TEMPORAL_NEGATIVE_TYPES = ("reverse", "shuffle")


class JointFlowPreparedWindowDataset(PreparedWindowDataset):
    """Prepared GM-100 windows with future observation latents and proprio history."""

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
        obs = self._camera_feature_stack(features, history_indices, "obs_features")
        future_obs = self._camera_feature_stack(features, future_indices, "future_obs_features")

        proprio = arrays["proprio"][t].astype(np.float32)
        proprio_history = arrays["proprio"][history_indices].astype(np.float32)
        action_chunk = arrays["action"][future_indices].astype(np.float32)
        norm_stats = self._norm_stats(source)
        if norm_stats is not None:
            proprio = normalize_array(proprio, norm_stats["proprio"])
            proprio_history = normalize_array(proprio_history, norm_stats["proprio"])
            action_chunk = normalize_array(action_chunk, norm_stats["action"])
        proprio = self._pad_last_dim(proprio, self.canonical_proprio_dim, "proprio")
        proprio_history = self._pad_last_dim(proprio_history, self.canonical_proprio_dim, "proprio_history")
        action_chunk = self._pad_last_dim(action_chunk, self.canonical_action_dim, "action")
        if "source_id" in self.labels:
            source_id = int(self.labels["source_id"][label_index])
        else:
            source_id = int(window.get("source_id", -1))

        sample = {
            "obs_features": torch.from_numpy(obs.astype(np.float32)),
            "future_obs_features": torch.from_numpy(future_obs.astype(np.float32)),
            "proprio": torch.from_numpy(proprio),
            "proprio_history": torch.from_numpy(proprio_history),
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


class MockJointFlowDataset(Dataset):
    """Toy joint-flow dataset for shape tests and CPU smoke runs."""

    def __init__(self, config: MockDatasetConfig | None = None) -> None:
        self.base = MockWindowDataset(config)
        c = self.base.config
        rng = np.random.default_rng(c.seed + 999)
        future_noise = rng.normal(
            size=(c.num_samples, c.horizon, c.cameras, c.feature_dim),
        ).astype(np.float32)
        action_signal = self.base.action_chunk.mean(axis=2)[:, :, None, None]
        self.future_obs_features = (
            self.base.obs_features[:, -1:, :, :] + 0.05 * future_noise + 0.03 * action_signal
        ).astype(np.float32)
        self.proprio_history = rng.normal(size=(c.num_samples, c.history, c.proprio_dim)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = dict(self.base[index])
        sample["future_obs_features"] = torch.from_numpy(self.future_obs_features[index])
        sample["proprio_history"] = torch.from_numpy(self.proprio_history[index])
        return sample


def make_joint_flow_loaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    data_cfg = config["data"]
    feature_cfg = config["features"]
    batch_size = int(data_cfg.get("batch_size", 8))

    if data_cfg.get("windows_dir"):
        return {
            split: DataLoader(
                JointFlowPreparedWindowDataset(
                    windows_dir=data_cfg["windows_dir"],
                    episodes_dir=data_cfg["episodes_dir"],
                    features_dir=data_cfg["features_dir"],
                    split=split,
                    feature_dim=int(feature_cfg["feature_dim"]),
                    norm_stats=data_cfg.get("norm_stats"),
                    prompt_features=data_cfg.get("prompt_features"),
                    prompt_feature_dim=(
                        int(data_cfg["prompt_feature_dim"])
                        if data_cfg.get("prompt_feature_dim") is not None
                        else None
                    ),
                    canonical_proprio_dim=(
                        int(data_cfg["canonical_proprio_dim"])
                        if data_cfg.get("canonical_proprio_dim") is not None
                        else None
                    ),
                    canonical_action_dim=(
                        int(data_cfg["canonical_action_dim"])
                        if data_cfg.get("canonical_action_dim") is not None
                        else None
                    ),
                    canonical_num_cameras=(
                        int(data_cfg["canonical_num_cameras"])
                        if data_cfg.get("canonical_num_cameras") is not None
                        else None
                    ),
                ),
                batch_size=batch_size,
                shuffle=(split == "train"),
                collate_fn=collate_batch,
            )
            for split in ("train", "val", "test")
        }

    base = MockDatasetConfig(
        num_samples=int(data_cfg.get("num_samples", 64)),
        history=int(data_cfg["history"]),
        horizon=int(data_cfg["horizon"]),
        cameras=int(data_cfg.get("cameras", 1)),
        feature_dim=int(feature_cfg["feature_dim"]),
        proprio_dim=int(data_cfg.get("proprio_dim", 14)),
        action_dim=int(data_cfg.get("action_dim", 14)),
        prompt_dim=int(data_cfg.get("prompt_feature_dim", data_cfg.get("prompt_dim", 512))),
        seed=int(config.get("seed", 42)),
    )
    return {
        "train": DataLoader(MockJointFlowDataset(base), batch_size=batch_size, shuffle=True, collate_fn=collate_batch),
        "val": DataLoader(
            MockJointFlowDataset(
                MockDatasetConfig(**{**base.__dict__, "num_samples": max(16, base.num_samples // 4), "seed": base.seed + 1})
            ),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        ),
        "test": DataLoader(
            MockJointFlowDataset(
                MockDatasetConfig(**{**base.__dict__, "num_samples": max(16, base.num_samples // 4), "seed": base.seed + 2})
            ),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_batch,
        ),
    }


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    timesteps = timesteps.float().reshape(-1, 1)
    half = dim // 2
    freq = torch.exp(
        torch.arange(half, device=timesteps.device, dtype=torch.float32)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    angles = timesteps * freq.unsqueeze(0)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class AdaLNTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, heads: int, dropout: float = 0.1, mlp_ratio: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 4))

    def _modulate(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma1, beta1, gamma2, beta2 = self.ada(cond).chunk(4, dim=-1)
        attn_in = self._modulate(self.norm1(x), gamma1, beta1)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self._modulate(self.norm2(x), gamma2, beta2))
        return x


class JointFlowDiT(nn.Module):
    CONDITION = 0
    NOISY = 1
    CLAMPED = 2

    def __init__(
        self,
        feature_dim: int = 768,
        prompt_dim: int = 768,
        proprio_dim: int = 14,
        action_dim: int = 14,
        hidden_dim: int = 192,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        mlp_ratio: int = 4,
        history: int = 4,
        horizon: int = 8,
        phi_tokens: int = 1,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.history = history
        self.horizon = horizon
        self.phi_tokens = phi_tokens

        self.prompt_proj = nn.Sequential(nn.LayerNorm(prompt_dim), nn.Linear(prompt_dim, hidden_dim), nn.GELU())
        self.obs_proj = nn.Linear(feature_dim, hidden_dim)
        self.future_obs_proj = nn.Linear(feature_dim, hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.phi_proj = nn.Linear(1, hidden_dim)

        self.modality_emb = nn.Embedding(6, hidden_dim)
        self.mask_emb = nn.Embedding(3, hidden_dim)
        self.time_emb = nn.Embedding(max(history, horizon, phi_tokens) + 1, hidden_dim)
        self.flow_time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [AdaLNTransformerBlock(hidden_dim, heads=heads, dropout=dropout, mlp_ratio=mlp_ratio) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.obs_head = nn.Linear(hidden_dim, feature_dim)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.phi_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _pool_features(features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 4:
            return features.float().mean(dim=2)
        if features.ndim == 3:
            return features.float()
        raise ValueError("Expected features with shape [B,T,C,D] or [B,T,D].")

    def _add_type_time(self, tokens: torch.Tensor, modality: int, mask: int, time_offset: int = 0) -> torch.Tensor:
        length = tokens.shape[1]
        device = tokens.device
        times = torch.arange(length, device=device).clamp(max=self.time_emb.num_embeddings - 1)
        times = (times + time_offset).clamp(max=self.time_emb.num_embeddings - 1)
        return (
            tokens
            + self.modality_emb(torch.full((length,), modality, device=device, dtype=torch.long)).unsqueeze(0)
            + self.mask_emb(torch.full((length,), mask, device=device, dtype=torch.long)).unsqueeze(0)
            + self.time_emb(times).unsqueeze(0)
        )

    def forward(
        self,
        obs_history: torch.Tensor,
        proprio_history: torch.Tensor,
        prompt_features: torch.Tensor,
        future_obs_noisy: torch.Tensor,
        action_noisy: torch.Tensor,
        phi_noisy: torch.Tensor,
        tau: torch.Tensor,
        action_is_condition: bool = False,
    ) -> dict[str, torch.Tensor]:
        obs_history = self._pool_features(obs_history)
        future_obs_noisy = self._pool_features(future_obs_noisy)
        if phi_noisy.ndim == 2:
            phi_noisy = phi_noisy.unsqueeze(-1)

        prompt = self.prompt_proj(prompt_features.float()).unsqueeze(1)
        hist_obs = self.obs_proj(obs_history.float())
        proprio = self.proprio_proj(proprio_history.float())
        future_obs = self.future_obs_proj(future_obs_noisy.float())
        action = self.action_proj(action_noisy.float())
        phi = self.phi_proj(phi_noisy.float())

        action_mask = self.CLAMPED if action_is_condition else self.NOISY
        pieces = [
            self._add_type_time(prompt, modality=0, mask=self.CONDITION),
            self._add_type_time(hist_obs, modality=1, mask=self.CONDITION),
            self._add_type_time(proprio, modality=2, mask=self.CONDITION),
            self._add_type_time(future_obs, modality=3, mask=self.NOISY),
            self._add_type_time(action, modality=4, mask=action_mask),
            self._add_type_time(phi, modality=5, mask=self.NOISY),
        ]
        lengths = [piece.shape[1] for piece in pieces]
        x = torch.cat(pieces, dim=1)
        cond = self.flow_time_mlp(sinusoidal_embedding(tau.reshape(-1), self.hidden_dim).to(x.dtype))
        for block in self.blocks:
            x = block(x, cond)
        x = self.final_norm(x)

        start = 0
        slices = []
        for length in lengths:
            slices.append(slice(start, start + length))
            start += length
        future_tokens = x[:, slices[3]]
        action_tokens = x[:, slices[4]]
        phi_tokens = x[:, slices[5]]
        return {
            "v_obs": self.obs_head(future_tokens),
            "v_action": self.action_head(action_tokens),
            "v_phi": self.phi_head(phi_tokens).squeeze(-1),
        }


class PhiOnlyFlowCritic(nn.Module):
    CONDITION = 0
    NOISY = 1
    CLAMPED = 2

    def __init__(
        self,
        feature_dim: int = 768,
        prompt_dim: int = 768,
        proprio_dim: int = 14,
        action_dim: int = 14,
        hidden_dim: int = 192,
        layers: int = 3,
        heads: int = 4,
        dropout: float = 0.1,
        mlp_ratio: int = 4,
        history: int = 4,
        horizon: int = 8,
        phi_tokens: int = 8,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.history = history
        self.horizon = horizon
        self.phi_tokens = phi_tokens

        self.prompt_proj = nn.Sequential(nn.LayerNorm(prompt_dim), nn.Linear(prompt_dim, hidden_dim), nn.GELU())
        self.obs_proj = nn.Linear(feature_dim, hidden_dim)
        self.proprio_proj = nn.Linear(proprio_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.phi_proj = nn.Linear(1, hidden_dim)

        self.modality_emb = nn.Embedding(5, hidden_dim)
        self.mask_emb = nn.Embedding(3, hidden_dim)
        self.time_emb = nn.Embedding(max(history, horizon, phi_tokens) + 1, hidden_dim)
        self.flow_time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [AdaLNTransformerBlock(hidden_dim, heads=heads, dropout=dropout, mlp_ratio=mlp_ratio) for _ in range(layers)]
        )
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.phi_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _pool_features(features: torch.Tensor) -> torch.Tensor:
        return JointFlowDiT._pool_features(features)

    def _add_type_time(self, tokens: torch.Tensor, modality: int, mask: int, time_offset: int = 0) -> torch.Tensor:
        length = tokens.shape[1]
        device = tokens.device
        times = torch.arange(length, device=device).clamp(max=self.time_emb.num_embeddings - 1)
        times = (times + time_offset).clamp(max=self.time_emb.num_embeddings - 1)
        return (
            tokens
            + self.modality_emb(torch.full((length,), modality, device=device, dtype=torch.long)).unsqueeze(0)
            + self.mask_emb(torch.full((length,), mask, device=device, dtype=torch.long)).unsqueeze(0)
            + self.time_emb(times).unsqueeze(0)
        )

    def forward(
        self,
        obs_history: torch.Tensor,
        proprio_history: torch.Tensor,
        prompt_features: torch.Tensor,
        future_obs_noisy: torch.Tensor,
        action_noisy: torch.Tensor,
        phi_noisy: torch.Tensor,
        tau: torch.Tensor,
        action_is_condition: bool = False,
    ) -> dict[str, torch.Tensor]:
        obs_history = self._pool_features(obs_history)
        future_obs_noisy = self._pool_features(future_obs_noisy)
        if phi_noisy.ndim == 2:
            phi_noisy = phi_noisy.unsqueeze(-1)

        prompt = self.prompt_proj(prompt_features.float()).unsqueeze(1)
        hist_obs = self.obs_proj(obs_history.float())
        proprio = self.proprio_proj(proprio_history.float())
        action = self.action_proj(action_noisy.float())
        phi = self.phi_proj(phi_noisy.float())

        action_mask = self.CLAMPED if action_is_condition else self.NOISY
        pieces = [
            self._add_type_time(prompt, modality=0, mask=self.CONDITION),
            self._add_type_time(hist_obs, modality=1, mask=self.CONDITION),
            self._add_type_time(proprio, modality=2, mask=self.CONDITION),
            self._add_type_time(action, modality=3, mask=action_mask),
            self._add_type_time(phi, modality=4, mask=self.NOISY),
        ]
        lengths = [piece.shape[1] for piece in pieces]
        x = torch.cat(pieces, dim=1)
        cond = self.flow_time_mlp(sinusoidal_embedding(tau.reshape(-1), self.hidden_dim).to(x.dtype))
        for block in self.blocks:
            x = block(x, cond)
        x = self.final_norm(x)

        start = 0
        slices = []
        for length in lengths:
            slices.append(slice(start, start + length))
            start += length
        phi_tokens = x[:, slices[4]]
        return {
            "v_obs": torch.zeros_like(future_obs_noisy),
            "v_action": torch.zeros_like(action_noisy),
            "v_phi": self.phi_head(phi_tokens).squeeze(-1),
        }


@dataclass(frozen=True)
class FlowBatch:
    future_obs_target: torch.Tensor
    action_target: torch.Tensor
    phi_target: torch.Tensor
    future_obs_noisy: torch.Tensor
    action_noisy: torch.Tensor
    phi_noisy: torch.Tensor
    tau: torch.Tensor
    v_obs_target: torch.Tensor
    v_action_target: torch.Tensor
    v_phi_target: torch.Tensor


def _randn_like(values: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.randn(values.shape, dtype=values.dtype, device=values.device, generator=generator)


def make_phi_target(
    batch: dict[str, torch.Tensor],
    phi_tokens: int = 1,
    mode: str = "delta_trajectory",
) -> torch.Tensor:
    delta_phi = batch["delta_phi"].float().reshape(-1, 1)
    if phi_tokens <= 1:
        return delta_phi

    if mode == "constant_delta":
        return delta_phi.expand(-1, phi_tokens)
    if mode == "delta_trajectory":
        ramp = torch.linspace(
            1.0 / float(phi_tokens),
            1.0,
            phi_tokens,
            device=delta_phi.device,
            dtype=delta_phi.dtype,
        ).reshape(1, -1)
        return delta_phi * ramp
    raise ValueError(f"Unknown phi target mode: {mode}")


def make_flow_batch(
    batch: dict[str, torch.Tensor],
    generator: torch.Generator | None = None,
    action_is_condition: bool = False,
    phi_tokens: int = 1,
    phi_target_mode: str = "delta_trajectory",
) -> FlowBatch:
    future_obs_target = JointFlowDiT._pool_features(batch["future_obs_features"]).float()
    action_target = batch["action_chunk"].float()
    phi_target = make_phi_target(batch, phi_tokens=phi_tokens, mode=phi_target_mode)

    tau = torch.rand((future_obs_target.shape[0],), device=future_obs_target.device, generator=generator)
    obs_tau = tau.reshape(-1, 1, 1)
    action_tau = tau.reshape(-1, 1, 1)
    phi_tau = tau.reshape(-1, 1)

    eps_obs = _randn_like(future_obs_target, generator=generator)
    eps_action = _randn_like(action_target, generator=generator)
    eps_phi = _randn_like(phi_target, generator=generator)
    future_obs_noisy = (1.0 - obs_tau) * eps_obs + obs_tau * future_obs_target
    if action_is_condition:
        action_noisy = action_target
        eps_action = torch.zeros_like(action_target)
    else:
        action_noisy = (1.0 - action_tau) * eps_action + action_tau * action_target
    phi_noisy = (1.0 - phi_tau) * eps_phi + phi_tau * phi_target

    return FlowBatch(
        future_obs_target=future_obs_target,
        action_target=action_target,
        phi_target=phi_target,
        future_obs_noisy=future_obs_noisy,
        action_noisy=action_noisy,
        phi_noisy=phi_noisy,
        tau=tau,
        v_obs_target=future_obs_target - eps_obs,
        v_action_target=action_target - eps_action,
        v_phi_target=phi_target - eps_phi,
    )


def build_joint_flow_model(config: dict[str, Any]) -> nn.Module:
    data_cfg = config["data"]
    feature_cfg = config["features"]
    model_cfg = config["model"]
    model_name = str(model_cfg.get("name", "lightweight_joint_flow_dit"))
    model_cls: type[nn.Module] = PhiOnlyFlowCritic if "phi_only" in model_name else JointFlowDiT
    return model_cls(
        feature_dim=int(feature_cfg["feature_dim"]),
        prompt_dim=int(data_cfg.get("prompt_feature_dim", data_cfg.get("prompt_dim", 768))),
        proprio_dim=int(data_cfg.get("proprio_dim", 14)),
        action_dim=int(data_cfg.get("action_dim", 14)),
        hidden_dim=int(model_cfg.get("hidden_dim", 192)),
        layers=int(model_cfg.get("transformer_layers", model_cfg.get("layers", 3))),
        heads=int(model_cfg.get("transformer_heads", model_cfg.get("heads", 4))),
        dropout=float(model_cfg.get("dropout", 0.1)),
        mlp_ratio=int(model_cfg.get("mlp_ratio", 4)),
        history=int(data_cfg.get("history", 4)),
        horizon=int(data_cfg.get("horizon", 8)),
        phi_tokens=int(model_cfg.get("phi_tokens", 1)),
    )


def phi_from_velocity(
    phi_noisy: torch.Tensor,
    v_phi: torch.Tensor,
    tau: torch.Tensor,
    clamp: bool = True,
    reduce: str = "last",
) -> torch.Tensor:
    if phi_noisy.ndim == 3:
        phi_noisy = phi_noisy.squeeze(-1)
    pred = phi_noisy + (1.0 - tau.reshape(-1, 1)) * v_phi
    if reduce == "last":
        score = pred[:, -1]
    elif reduce == "mean":
        score = pred.mean(dim=1)
    else:
        raise ValueError(f"Unknown phi reduce mode: {reduce}")
    return score.clamp(0.0, 1.0) if clamp else score


def score_action(
    model: JointFlowDiT,
    batch: dict[str, torch.Tensor],
    action_chunk: torch.Tensor | None = None,
    tau_value: float = 0.0,
    clamp: bool = True,
    denoise_steps: int = 1,
    phi_tokens: int = 1,
    phi_reduce: str = "last",
    future_obs_init: str = "zero",
) -> torch.Tensor:
    batch_size = int(batch["action_chunk"].shape[0])
    future_obs = JointFlowDiT._pool_features(batch["future_obs_features"]).float()
    action = batch["action_chunk"].float() if action_chunk is None else action_chunk.float()
    if future_obs_init == "zero":
        future_obs_state = torch.zeros_like(future_obs)
    elif future_obs_init == "target":
        future_obs_state = future_obs
    else:
        raise ValueError(f"Unknown future obs init: {future_obs_init}")
    phi_state = torch.zeros((batch_size, max(1, phi_tokens)), device=action.device, dtype=action.dtype)

    steps = max(1, int(denoise_steps))
    if steps == 1:
        tau = torch.full((batch_size,), float(tau_value), device=action.device, dtype=action.dtype)
        outputs = model(
            batch["obs_features"],
            batch["proprio_history"],
            batch["prompt_features"],
            future_obs_state,
            action,
            phi_state,
            tau,
            action_is_condition=True,
        )
        return phi_from_velocity(phi_state, outputs["v_phi"], tau, clamp=clamp, reduce=phi_reduce)

    dt = 1.0 / float(steps)
    for step in range(steps):
        tau = torch.full((batch_size,), step * dt, device=action.device, dtype=action.dtype)
        outputs = model(
            batch["obs_features"],
            batch["proprio_history"],
            batch["prompt_features"],
            future_obs_state,
            action,
            phi_state,
            tau,
            action_is_condition=True,
        )
        future_obs_state = future_obs_state + dt * outputs["v_obs"]
        phi_state = phi_state + dt * outputs["v_phi"]
    if phi_reduce == "last":
        score = phi_state[:, -1]
    elif phi_reduce == "mean":
        score = phi_state.mean(dim=1)
    else:
        raise ValueError(f"Unknown phi reduce mode: {phi_reduce}")
    return score.clamp(0.0, 1.0) if clamp else score


def joint_flow_loss(
    outputs: dict[str, torch.Tensor],
    flow: FlowBatch,
    loss_cfg: dict[str, Any],
    action_is_condition: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    obs_loss = F.mse_loss(outputs["v_obs"], flow.v_obs_target)
    action_loss = F.mse_loss(outputs["v_action"], flow.v_action_target) if not action_is_condition else outputs["v_action"].sum() * 0.0
    phi_loss = F.mse_loss(outputs["v_phi"], flow.v_phi_target)
    total = (
        float(loss_cfg.get("obs_weight", 1.0)) * obs_loss
        + float(loss_cfg.get("action_weight", 1.0)) * action_loss
        + float(loss_cfg.get("phi_weight", 5.0)) * phi_loss
    )
    return total, {
        "obs_flow_loss": float(obs_loss.detach().cpu()),
        "action_flow_loss": float(action_loss.detach().cpu()),
        "phi_flow_loss": float(phi_loss.detach().cpu()),
    }


def cf_ranking_loss(pos_phi: torch.Tensor, neg_phi: torch.Tensor, margin: float) -> torch.Tensor:
    return -F.logsigmoid(pos_phi.reshape(-1) - neg_phi.reshape(-1) - margin).mean()


def training_negative_types(config: dict[str, Any]) -> list[str]:
    negative_cfg = config.get("negatives", {})
    train_types = parse_negative_types(negative_cfg.get("train_types"))
    if train_types:
        return train_types
    return expand_negative_types(negative_cfg)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def grouped_negative_metrics(
    pos_delta_phi: torch.Tensor,
    neg_delta_phi: torch.Tensor,
    negative_types: list[str],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    type_array = np.asarray(negative_types)
    pos = pos_delta_phi.detach().cpu()
    neg = neg_delta_phi.detach().cpu()
    for prefix, group_types in (
        ("coarse_action_cf", COARSE_NEGATIVE_TYPES),
        ("temporal_diagnostic", TEMPORAL_NEGATIVE_TYPES),
    ):
        mask = np.isin(type_array, list(group_types))
        if not np.any(mask):
            continue
        mask_tensor = torch.from_numpy(mask).to(dtype=torch.bool)
        group_pos = pos[mask_tensor]
        group_neg = neg[mask_tensor]
        metrics[f"{prefix}_ranking_acc"] = float(tie_aware_ranking(group_pos, group_neg))
        metrics[f"{prefix}_mean_margin"] = float(torch.mean(group_pos - group_neg).item())
    return metrics


def _safe_metric_name(value: object) -> str:
    text = str(value) if str(value) else "unknown"
    cleaned = "".join(char if char.isalnum() else "_" for char in text.lower())
    return "_".join(part for part in cleaned.split("_") if part) or "unknown"


def source_id_to_name(config: dict[str, Any]) -> dict[int, str]:
    data_cfg = config.get("data", {})
    windows_dir = data_cfg.get("windows_dir")
    if not windows_dir:
        return {}
    index_path = Path(windows_dir) / "index.json"
    if not index_path.exists():
        return {}
    try:
        with index_path.open("r", encoding="utf-8") as handle:
            index = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}
    source_to_id = index.get("source_to_id")
    if not isinstance(source_to_id, dict):
        return {}
    return {int(source_id): str(source) for source, source_id in source_to_id.items()}


def _source_metric_prefix(source_id: int, names: dict[int, str]) -> str:
    source_name = names.get(int(source_id), f"id_{int(source_id)}")
    return f"source_{_safe_metric_name(source_name)}"


def source_stratified_metrics(
    pred_delta_phi: torch.Tensor,
    target_delta_phi: torch.Tensor,
    source_ids: torch.Tensor,
    source_names: dict[int, str] | None = None,
    pos_delta_phi: torch.Tensor | None = None,
    neg_delta_phi: torch.Tensor | None = None,
    negative_types: list[str] | None = None,
    negative_source_ids: torch.Tensor | None = None,
) -> dict[str, float]:
    source_ids = source_ids.detach().cpu().reshape(-1)
    if source_ids.numel() == 0:
        return {}
    source_names = source_names or {}
    metrics: dict[str, float] = {}
    for raw_source_id in sorted({int(item) for item in source_ids.tolist()}):
        mask = source_ids == raw_source_id
        prefix = _source_metric_prefix(raw_source_id, source_names)
        source_metrics = compute_metrics(pred_delta_phi[mask], target_delta_phi[mask])
        for key, value in source_metrics.items():
            metrics[f"{prefix}_{key}"] = value
        metrics[f"{prefix}_num_windows"] = float(int(mask.sum().item()))

    if (
        pos_delta_phi is None
        or neg_delta_phi is None
        or negative_types is None
        or negative_source_ids is None
        or len(negative_types) == 0
    ):
        return metrics

    negative_source_ids = negative_source_ids.detach().cpu().reshape(-1)
    if int(negative_source_ids.numel()) != len(negative_types):
        raise ValueError("negative_source_ids and negative_types must have matching lengths.")
    for raw_source_id in sorted({int(item) for item in negative_source_ids.tolist()}):
        mask = negative_source_ids == raw_source_id
        prefix = _source_metric_prefix(raw_source_id, source_names)
        source_pos = pos_delta_phi[mask]
        source_neg = neg_delta_phi[mask]
        source_types = [negative_type for negative_type, keep in zip(negative_types, mask.tolist(), strict=True) if keep]
        metrics[f"{prefix}_all_negatives_tie_aware_ranking_acc"] = float(tie_aware_ranking(source_pos, source_neg))
        metrics[f"{prefix}_all_negatives_mean_margin"] = float(torch.mean(source_pos - source_neg).item())
        for key, value in summarize_by_type(source_pos, source_neg, source_types).items():
            metrics[f"{prefix}_{key}"] = value
        for key, value in grouped_negative_metrics(source_pos, source_neg, source_types).items():
            metrics[f"{prefix}_{key}"] = value
    return metrics


def joint_flow_runtime_options(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config.get("model", {})
    score_cfg = config.get("score", {})
    return {
        "phi_tokens": int(model_cfg.get("phi_tokens", 1)),
        "phi_target_mode": str(model_cfg.get("phi_target_mode", "delta_trajectory")),
        "score_denoise_steps": int(score_cfg.get("denoise_steps", 1)),
        "train_score_denoise_steps": int(score_cfg.get("train_denoise_steps", score_cfg.get("denoise_steps", 1))),
        "phi_reduce": str(score_cfg.get("phi_reduce", "last")),
        "future_obs_init": str(score_cfg.get("future_obs_init", "zero")),
    }


@torch.no_grad()
def evaluate_joint_flow(
    model: JointFlowDiT,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    split: str,
    output_dir: str | Path | None = None,
) -> dict[str, float]:
    model.eval()
    eval_cfg = config.get("eval", {})
    negative_types = parse_negative_types(
        eval_cfg.get("negative_types", "zero,reverse,shuffle,wrong_arm,scaled_0.25,scaled_1.75")
    )
    runtime = joint_flow_runtime_options(config)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(config.get("seed", 42)) + 1000 + {"train": 0, "val": 1, "test": 2}.get(split, 3))

    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    obs_losses: list[float] = []
    action_losses: list[float] = []
    phi_losses: list[float] = []
    action_mses: list[float] = []
    obs_mses: list[float] = []
    critic_obs_losses: list[float] = []
    critic_phi_losses: list[float] = []
    critic_obs_mses: list[float] = []
    sensitivity_rows: list[dict[str, float | int | str]] = []
    all_pos: list[torch.Tensor] = []
    all_neg: list[torch.Tensor] = []
    all_types: list[str] = []
    all_source_ids: list[torch.Tensor] = []
    all_negative_source_ids: list[torch.Tensor] = []
    offset = 0
    source_names = source_id_to_name(config)

    for batch in loader:
        batch = batch_to_device(batch, device)
        pred_phi = score_action(
            model,
            batch,
            clamp=True,
            denoise_steps=runtime["score_denoise_steps"],
            phi_tokens=runtime["phi_tokens"],
            phi_reduce=runtime["phi_reduce"],
            future_obs_init=runtime["future_obs_init"],
        )
        target_phi = batch["delta_phi"].float().reshape(-1)
        preds.append(pred_phi.cpu())
        targets.append(target_phi.cpu())
        if "source_id" in batch:
            all_source_ids.append(batch["source_id"].detach().cpu().reshape(-1))

        flow = make_flow_batch(
            batch,
            generator=generator,
            action_is_condition=False,
            phi_tokens=runtime["phi_tokens"],
            phi_target_mode=runtime["phi_target_mode"],
        )
        outputs = model(
            batch["obs_features"],
            batch["proprio_history"],
            batch["prompt_features"],
            flow.future_obs_noisy,
            flow.action_noisy,
            flow.phi_noisy,
            flow.tau,
            action_is_condition=False,
        )
        loss_cfg = config.get("loss", {})
        _, parts = joint_flow_loss(outputs, flow, loss_cfg, action_is_condition=False)
        obs_losses.append(parts["obs_flow_loss"])
        action_losses.append(parts["action_flow_loss"])
        phi_losses.append(parts["phi_flow_loss"])
        pred_action_y0 = flow.action_noisy + (1.0 - flow.tau.reshape(-1, 1, 1)) * outputs["v_action"]
        pred_obs_y0 = flow.future_obs_noisy + (1.0 - flow.tau.reshape(-1, 1, 1)) * outputs["v_obs"]
        action_mses.append(float(F.mse_loss(pred_action_y0, flow.action_target).detach().cpu()))
        obs_mses.append(float(F.mse_loss(pred_obs_y0, flow.future_obs_target).detach().cpu()))

        critic_flow = make_flow_batch(
            batch,
            generator=generator,
            action_is_condition=True,
            phi_tokens=runtime["phi_tokens"],
            phi_target_mode=runtime["phi_target_mode"],
        )
        critic_outputs = model(
            batch["obs_features"],
            batch["proprio_history"],
            batch["prompt_features"],
            critic_flow.future_obs_noisy,
            critic_flow.action_noisy,
            critic_flow.phi_noisy,
            critic_flow.tau,
            action_is_condition=True,
        )
        _, critic_parts = joint_flow_loss(critic_outputs, critic_flow, loss_cfg, action_is_condition=True)
        critic_obs_losses.append(critic_parts["obs_flow_loss"])
        critic_phi_losses.append(critic_parts["phi_flow_loss"])
        pred_critic_obs_y0 = (
            critic_flow.future_obs_noisy
            + (1.0 - critic_flow.tau.reshape(-1, 1, 1)) * critic_outputs["v_obs"]
        )
        critic_obs_mses.append(
            float(F.mse_loss(pred_critic_obs_y0, critic_flow.future_obs_target).detach().cpu())
        )

        pos_scores = score_action(
            model,
            batch,
            action_chunk=batch["action_chunk"],
            clamp=True,
            denoise_steps=runtime["score_denoise_steps"],
            phi_tokens=runtime["phi_tokens"],
            phi_reduce=runtime["phi_reduce"],
            future_obs_init=runtime["future_obs_init"],
        ).cpu()
        for negative_type in negative_types:
            try:
                neg_action = make_negative_action_tensor(
                    batch["action_chunk"],
                    kind=negative_type,
                    stage_id=batch.get("stage_id"),
                    source_id=batch.get("source_id"),
                )
            except ValueError:
                continue
            pos = pos_scores
            neg = score_action(
                model,
                batch,
                action_chunk=neg_action,
                clamp=True,
                denoise_steps=runtime["score_denoise_steps"],
                phi_tokens=runtime["phi_tokens"],
                phi_reduce=runtime["phi_reduce"],
                future_obs_init=runtime["future_obs_init"],
            ).cpu()
            margin = pos - neg
            all_pos.append(pos)
            all_neg.append(neg)
            all_types.extend([negative_type] * int(pos.shape[0]))
            if "source_id" in batch:
                all_negative_source_ids.append(batch["source_id"].detach().cpu().reshape(-1))
            for i in range(int(pos.shape[0])):
                sensitivity_rows.append(
                    {
                        "index": offset + i,
                        "negative_type": negative_type,
                        "pos_delta_phi": float(pos[i]),
                        "neg_delta_phi": float(neg[i]),
                        "margin": float(margin[i]),
                        "is_correct": int(pos[i] > neg[i]),
                    }
                )
        offset += int(target_phi.shape[0])

    pred = torch.cat(preds)
    target = torch.cat(targets)
    metrics = compute_metrics(pred, target)
    metrics.update(
        {
            "obs_flow_mse": _mean(obs_losses),
            "action_flow_mse": _mean(action_losses),
            "phi_flow_mse": _mean(phi_losses),
            "action_y0_mse": _mean(action_mses),
            "future_obs_y0_mse": _mean(obs_mses),
            "predictor_obs_flow_mse": _mean(obs_losses),
            "predictor_action_flow_mse": _mean(action_losses),
            "predictor_phi_flow_mse": _mean(phi_losses),
            "predictor_action_y0_mse": _mean(action_mses),
            "predictor_future_obs_y0_mse": _mean(obs_mses),
            "critic_delta_phi_mae": metrics["delta_phi_mae"],
            "critic_delta_phi_rmse": metrics["delta_phi_rmse"],
            "critic_obs_flow_mse": _mean(critic_obs_losses),
            "critic_phi_flow_mse": _mean(critic_phi_losses),
            "critic_future_obs_y0_mse": _mean(critic_obs_mses),
        }
    )
    if all_pos and all_neg:
        pos_all = torch.cat(all_pos)
        neg_all = torch.cat(all_neg)
        metrics["all_negatives_tie_aware_ranking_acc"] = float(tie_aware_ranking(pos_all, neg_all))
        metrics["all_negatives_mean_margin"] = float(torch.mean(pos_all - neg_all).item())
        metrics.update(summarize_by_type(pos_all, neg_all, all_types))
        metrics.update(grouped_negative_metrics(pos_all, neg_all, all_types))
        if all_source_ids and all_negative_source_ids:
            metrics.update(
                source_stratified_metrics(
                    pred,
                    target,
                    torch.cat(all_source_ids),
                    source_names=source_names,
                    pos_delta_phi=pos_all,
                    neg_delta_phi=neg_all,
                    negative_types=all_types,
                    negative_source_ids=torch.cat(all_negative_source_ids),
                )
            )
    elif all_source_ids:
        metrics.update(source_stratified_metrics(pred, target, torch.cat(all_source_ids), source_names=source_names))

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for i, (pred_value, target_value) in enumerate(zip(pred.tolist(), target.tolist(), strict=True)):
                handle.write(
                    json.dumps(
                        {"index": i, "pred_delta_phi": pred_value, "target_delta_phi": target_value},
                        sort_keys=True,
                    )
                    + "\n"
                )
        with (out / "action_sensitivity.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["index", "negative_type", "pos_delta_phi", "neg_delta_phi", "margin", "is_correct"],
            )
            writer.writeheader()
            writer.writerows(sensitivity_rows)
        with (out / "metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
    return metrics


def train_joint_flow(config: dict[str, Any]) -> dict[str, float]:
    set_seed(int(config.get("seed", 42)))
    device = torch.device(config.get("device", "cpu"))
    output_dir = Path(config.get("output_dir", "outputs")) / EXPERIMENT
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = make_joint_flow_loaders(config)
    model = build_joint_flow_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optim"]["lr"]),
        weight_decay=float(config["optim"].get("weight_decay", 0.0)),
    )
    max_epochs = int(config["train"].get("max_epochs", 2))
    save_best_by = str(config.get("train", {}).get("save_best_by", "val/delta_phi_mae"))
    loss_cfg = config.get("loss", {})
    cf_weight = float(loss_cfg.get("counterfactual_weight", 0.05))
    critic_flow_weight = float(loss_cfg.get("critic_flow_weight", 0.0))
    margin = float(loss_cfg.get("margin", 0.03))
    action_condition_prob = float(config.get("train", {}).get("action_condition_prob", 0.5))
    cf_negatives_per_batch_raw = config.get("train", {}).get("cf_negatives_per_batch", 1)
    negative_types = training_negative_types(config)
    runtime = joint_flow_runtime_options(config)
    rng = np.random.default_rng(int(config.get("seed", 42)))
    torch_generator = torch.Generator(device=device)
    torch_generator.manual_seed(int(config.get("seed", 42)) + 17)

    best_score = -float("inf")
    best_metrics: dict[str, float] = {}
    history: list[dict[str, float]] = []
    for epoch in range(max_epochs):
        model.train()
        losses: list[float] = []
        obs_parts: list[float] = []
        action_parts: list[float] = []
        phi_parts: list[float] = []
        cf_parts: list[float] = []
        critic_parts: list[float] = []
        for batch in loaders["train"]:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            action_is_condition = bool(rng.random() < action_condition_prob)
            flow = make_flow_batch(
                batch,
                generator=torch_generator,
                action_is_condition=action_is_condition,
                phi_tokens=runtime["phi_tokens"],
                phi_target_mode=runtime["phi_target_mode"],
            )
            outputs = model(
                batch["obs_features"],
                batch["proprio_history"],
                batch["prompt_features"],
                flow.future_obs_noisy,
                flow.action_noisy,
                flow.phi_noisy,
                flow.tau,
                action_is_condition=action_is_condition,
            )
            loss, parts = joint_flow_loss(outputs, flow, loss_cfg, action_is_condition=action_is_condition)
            critic_loss_value = torch.tensor(0.0, device=device)
            if critic_flow_weight > 0:
                critic_flow = make_flow_batch(
                    batch,
                    generator=torch_generator,
                    action_is_condition=True,
                    phi_tokens=runtime["phi_tokens"],
                    phi_target_mode=runtime["phi_target_mode"],
                )
                critic_outputs = model(
                    batch["obs_features"],
                    batch["proprio_history"],
                    batch["prompt_features"],
                    critic_flow.future_obs_noisy,
                    critic_flow.action_noisy,
                    critic_flow.phi_noisy,
                    critic_flow.tau,
                    action_is_condition=True,
                )
                critic_loss_value, _ = joint_flow_loss(
                    critic_outputs,
                    critic_flow,
                    loss_cfg,
                    action_is_condition=True,
                )
                loss = loss + critic_flow_weight * critic_loss_value

            cf_loss_value = torch.tensor(0.0, device=device)
            if cf_weight > 0 and negative_types:
                if cf_negatives_per_batch_raw == "all":
                    selected_types = list(negative_types)
                else:
                    cf_count = max(1, int(cf_negatives_per_batch_raw))
                    selected_indices = rng.choice(
                        len(negative_types),
                        size=min(cf_count, len(negative_types)),
                        replace=False,
                    )
                    selected_types = [str(negative_types[int(index)]) for index in np.asarray(selected_indices).reshape(-1)]
                pos_phi = score_action(
                    model,
                    batch,
                    action_chunk=batch["action_chunk"],
                    clamp=False,
                    denoise_steps=runtime["train_score_denoise_steps"],
                    phi_tokens=runtime["phi_tokens"],
                    phi_reduce=runtime["phi_reduce"],
                    future_obs_init=runtime["future_obs_init"],
                )
                cf_losses = []
                for negative_kind in selected_types:
                    neg_action = make_negative_action_tensor(
                        batch["action_chunk"],
                        kind=negative_kind,
                        stage_id=batch.get("stage_id"),
                        source_id=batch.get("source_id"),
                        generator=torch_generator,
                    )
                    neg_phi = score_action(
                        model,
                        batch,
                        action_chunk=neg_action,
                        clamp=False,
                        denoise_steps=runtime["train_score_denoise_steps"],
                        phi_tokens=runtime["phi_tokens"],
                        phi_reduce=runtime["phi_reduce"],
                        future_obs_init=runtime["future_obs_init"],
                    )
                    cf_losses.append(cf_ranking_loss(pos_phi, neg_phi, margin=margin))
                cf_loss_value = torch.stack(cf_losses).mean()
                loss = loss + cf_weight * cf_loss_value

            loss.backward()
            grad_clip = float(config["optim"].get("grad_clip_norm", 0.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            obs_parts.append(parts["obs_flow_loss"])
            action_parts.append(parts["action_flow_loss"])
            phi_parts.append(parts["phi_flow_loss"])
            cf_parts.append(float(cf_loss_value.detach().cpu()))
            critic_parts.append(float(critic_loss_value.detach().cpu()))

        val_metrics = evaluate_joint_flow(model, loaders["val"], config, device, split="val")
        val_metrics.update(
            {
                "epoch": float(epoch),
                "train_loss": _mean(losses),
                "train_obs_flow_loss": _mean(obs_parts),
                "train_action_flow_loss": _mean(action_parts),
                "train_phi_flow_loss": _mean(phi_parts),
                "train_cf_loss": _mean(cf_parts),
                "train_critic_flow_loss": _mean(critic_parts),
            }
        )
        history.append(dict(val_metrics))
        score = score_for_checkpoint(val_metrics, save_best_by)
        if score > best_score:
            best_score = score
            best_metrics = dict(val_metrics)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "experiment": EXPERIMENT,
                    "metrics": best_metrics,
                },
                output_dir / "best.pt",
            )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(best_metrics, handle, indent=2, sort_keys=True)
    with (output_dir / "history.jsonl").open("w", encoding="utf-8") as handle:
        for row in history:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_manifest(
        output_dir / "manifest.json",
        kind="train",
        config=config,
        metrics=best_metrics,
        experiment=EXPERIMENT,
        checkpoint=str(output_dir / "best.pt"),
        repo_root=Path(__file__).resolve().parents[1],
    )
    print(json.dumps(best_metrics, indent=2, sort_keys=True))
    return best_metrics


def load_joint_flow_checkpoint(checkpoint: str | Path, device: torch.device) -> tuple[JointFlowDiT, dict[str, Any]]:
    checkpoint_path = Path(checkpoint)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = payload["config"]
    model = build_joint_flow_model(config).to(device)
    model.load_state_dict(payload["model_state"])
    return model, config


def plot_joint_flow_outputs(run_dir: str | Path) -> list[Path]:
    run_dir = Path(run_dir)
    eval_dir = run_dir / "eval_test"
    pred_path = eval_dir / "predictions.jsonl"
    sensitivity_path = eval_dir / "action_sensitivity.csv"
    history_path = run_dir / "history.jsonl"
    plot_dir = run_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    preds: list[float] = []
    targets: list[float] = []
    if pred_path.exists():
        with pred_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                preds.append(float(row["pred_delta_phi"]))
                targets.append(float(row["target_delta_phi"]))
        fig, ax = plt.subplots(figsize=(4.8, 4.5))
        ax.scatter(targets, preds, s=8, alpha=0.35)
        ax.plot([0, 1], [0, 1], color="black", linewidth=1)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("target DeltaPhi")
        ax.set_ylabel("predicted DeltaPhi")
        fig.tight_layout()
        path = plot_dir / "delta_phi_scatter.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(preds, bins=25, alpha=0.65, label="pred")
        ax.hist(targets, bins=25, alpha=0.65, label="target")
        ax.set_xlabel("DeltaPhi")
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        path = plot_dir / "delta_phi_hist.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    if sensitivity_path.exists():
        by_type: dict[str, list[float]] = {}
        with sensitivity_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                by_type.setdefault(row["negative_type"], []).append(float(row["margin"]))
        fig, ax = plt.subplots(figsize=(7, 4))
        for negative_type, values in sorted(by_type.items()):
            ax.hist(values, bins=25, alpha=0.45, label=negative_type)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xlabel("pred DeltaPhi(pos) - pred DeltaPhi(neg)")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = plot_dir / "action_margin_hist.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

        fig, ax = plt.subplots(figsize=(7, 4))
        labels = sorted(by_type)
        ranking = [
            float(np.mean((np.asarray(by_type[label]) > 0.0).astype(np.float64) + 0.5 * np.isclose(by_type[label], 0.0)))
            for label in labels
        ]
        ax.bar(labels, ranking, color="#4E79A7", edgecolor="#26394F", linewidth=0.8)
        ax.axhline(0.5, color="black", linewidth=1)
        ax.set_ylim(0, 1)
        ax.set_ylabel("tie-aware ranking accuracy")
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        path = plot_dir / "per_negative_ranking.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    if history_path.exists():
        rows = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if rows:
            fig, ax = plt.subplots(figsize=(7, 4))
            epochs = [row["epoch"] for row in rows]
            for key in ("train_loss", "delta_phi_mae", "delta_phi_rmse"):
                if key in rows[0]:
                    ax.plot(epochs, [row[key] for row in rows], marker="o", label=key)
            ax.set_xlabel("epoch")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            path = plot_dir / "training_curves.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            written.append(path)
    return written


def write_experiment_report(run_dir: str | Path, test_metrics: dict[str, float], figures: list[Path]) -> Path:
    run_dir = Path(run_dir)
    lines = [
        "# MVP1 Joint Flow PP-WAM Smoke Report",
        "",
        "- Experiment: `mvp1_joint_flow`",
        "- Backbone: lightweight typed-token DiT denoiser",
        "- Inputs: prompt, history observation latents, proprio history, noisy future obs/action/potential tokens",
        "- Outputs: flow velocities for future obs latent, action chunk, and DeltaPhi potential",
        "",
        "## Test Metrics",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in sorted(test_metrics):
        if isinstance(test_metrics[key], (float, int)):
            lines.append(f"| `{key}` | {float(test_metrics[key]):.6f} |")
    lines.extend(["", "## Figures", ""])
    for figure in figures:
        lines.append(f"- `{figure}`")
    path = run_dir / "experiment_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_full_joint_flow(config: dict[str, Any]) -> dict[str, float]:
    train_joint_flow(config)
    run_dir = Path(config.get("output_dir", "outputs")) / EXPERIMENT
    device = torch.device(config.get("device", "cpu"))
    model, loaded_config = load_joint_flow_checkpoint(run_dir / "best.pt", device)
    loaders = make_joint_flow_loaders(loaded_config)
    test_dir = run_dir / "eval_test"
    test_metrics = evaluate_joint_flow(model, loaders["test"], loaded_config, device, split="test", output_dir=test_dir)
    write_manifest(
        test_dir / "manifest.json",
        kind="eval",
        config=loaded_config,
        metrics=test_metrics,
        experiment=EXPERIMENT,
        checkpoint=str(run_dir / "best.pt"),
        split="test",
        repo_root=Path(__file__).resolve().parents[1],
    )
    figures = plot_joint_flow_outputs(run_dir)
    write_experiment_report(run_dir, test_metrics, figures)
    print(json.dumps(test_metrics, indent=2, sort_keys=True))
    return test_metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train/evaluate MVP1 latent joint flow PP-WAM.")
    parser.add_argument("--config", default="configs/gm100/joint_flow_cf1p0.yaml")
    parser.add_argument("overrides", nargs="*", help="Optional key=value overrides.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    run_full_joint_flow(config)


if __name__ == "__main__":
    main()
