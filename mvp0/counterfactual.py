from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch


NegativeKind = Literal["zero", "reverse", "shuffle", "wrong_arm", "scaled"]


@dataclass(frozen=True)
class ActionRange:
    low: np.ndarray
    high: np.ndarray


@dataclass(frozen=True)
class PairedBatch:
    positive: dict[str, torch.Tensor]
    negative: dict[str, torch.Tensor]
    negative_type: tuple[str, ...]


def _as_action_chunk(action_chunk: np.ndarray) -> np.ndarray:
    array = np.asarray(action_chunk)
    if array.ndim != 2:
        raise ValueError("action_chunk must have shape [H, A].")
    return array


def wrong_arm_swap(action_chunk: np.ndarray) -> np.ndarray:
    action = _as_action_chunk(action_chunk)
    action_dim = action.shape[1]
    if action_dim % 2 != 0:
        raise ValueError("wrong_arm requires an even action dimension.")
    half = action_dim // 2
    return np.concatenate([action[:, half:], action[:, :half]], axis=1)


def scaled_action(
    action_chunk: np.ndarray,
    scale: float,
    action_range: ActionRange | None = None,
) -> np.ndarray:
    action = _as_action_chunk(action_chunk)
    scaled = action * scale
    if action_range is not None:
        scaled = np.clip(scaled, action_range.low, action_range.high)
    return scaled


def make_simple_negative(
    action_chunk: np.ndarray,
    kind: NegativeKind,
    rng: np.random.Generator | None = None,
    replacement_chunk: np.ndarray | None = None,
    scale: float = 0.25,
    action_range: ActionRange | None = None,
) -> np.ndarray:
    action = _as_action_chunk(action_chunk)

    if kind == "zero":
        return np.zeros_like(action)
    if kind == "reverse":
        return action[::-1].copy()
    if kind == "shuffle":
        if replacement_chunk is not None:
            replacement = _as_action_chunk(replacement_chunk)
            if replacement.shape != action.shape:
                raise ValueError("replacement_chunk shape must match action_chunk.")
            return replacement.copy()
        if rng is None:
            rng = np.random.default_rng()
        indices = rng.permutation(action.shape[0])
        return action[indices].copy()
    if kind == "wrong_arm":
        return wrong_arm_swap(action)
    if kind == "scaled":
        return scaled_action(action, scale=scale, action_range=action_range)

    raise ValueError(f"Unknown negative kind: {kind}")


def _clone_tensor_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() if torch.is_tensor(value) else value for key, value in batch.items()}


def make_negative_action_tensor(
    action_chunk: torch.Tensor,
    kind: NegativeKind,
    stage_id: torch.Tensor | None = None,
    scale: float = 0.25,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if action_chunk.ndim != 3:
        raise ValueError("action_chunk must have shape [B, H, A].")

    if kind == "zero":
        return torch.zeros_like(action_chunk)
    if kind == "reverse":
        return torch.flip(action_chunk, dims=(1,))
    if kind == "wrong_arm":
        action_dim = action_chunk.shape[-1]
        if action_dim % 2 != 0:
            raise ValueError("wrong_arm requires an even action dimension.")
        half = action_dim // 2
        return torch.cat([action_chunk[..., half:], action_chunk[..., :half]], dim=-1)
    if kind == "scaled":
        return torch.clamp(action_chunk * scale, min=-1.0, max=1.0)
    if kind == "shuffle":
        batch_size = action_chunk.shape[0]
        if stage_id is None:
            indices = torch.randperm(batch_size, generator=generator, device=action_chunk.device)
            return action_chunk[indices].clone()

        shuffled = action_chunk.clone()
        for i in range(batch_size):
            same_stage = torch.nonzero(stage_id == stage_id[i], as_tuple=False).flatten()
            same_stage = same_stage[same_stage != i]
            if len(same_stage) == 0:
                j = (i + 1) % batch_size
            else:
                pick = torch.randint(
                    low=0,
                    high=len(same_stage),
                    size=(1,),
                    generator=generator,
                    device=same_stage.device,
                )
                j = int(same_stage[pick].item())
            shuffled[i] = action_chunk[j]
        return shuffled

    raise ValueError(f"Unknown negative kind: {kind}")


def make_negative_batch(
    batch: dict[str, torch.Tensor],
    kind: NegativeKind = "zero",
    scale: float = 0.25,
    generator: torch.Generator | None = None,
) -> PairedBatch:
    if "action_chunk" not in batch:
        raise KeyError("batch must contain action_chunk.")

    positive = _clone_tensor_batch(batch)
    negative = _clone_tensor_batch(batch)
    negative["action_chunk"] = make_negative_action_tensor(
        batch["action_chunk"],
        kind=kind,
        stage_id=batch.get("stage_id"),
        scale=scale,
        generator=generator,
    )
    batch_size = int(batch["action_chunk"].shape[0])
    return PairedBatch(
        positive=positive,
        negative=negative,
        negative_type=tuple(kind for _ in range(batch_size)),
    )
