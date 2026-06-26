from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


NegativeKind = Literal["zero", "reverse", "shuffle", "wrong_arm", "scaled"]


@dataclass(frozen=True)
class ActionRange:
    low: np.ndarray
    high: np.ndarray


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

