from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ppwam.labels import compute_window_label, compute_window_label_from_potential
from ppwam.schemas import EpisodeSpec, WindowRecord


def split_episodes(
    episode_ids: Sequence[str],
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> dict[str, list[str]]:
    if not episode_ids:
        raise ValueError("episode_ids must not be empty.")
    if len(ratios) != 3:
        raise ValueError("ratios must be a 3-tuple for train/val/test.")
    if any(ratio < 0 for ratio in ratios):
        raise ValueError("ratios must be non-negative.")
    total = sum(ratios)
    if total <= 0:
        raise ValueError("At least one split ratio must be positive.")

    ids = list(episode_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)

    train_ratio, val_ratio, _ = [ratio / total for ratio in ratios]
    train_end = int(round(len(ids) * train_ratio))
    val_end = train_end + int(round(len(ids) * val_ratio))
    train_end = min(train_end, len(ids))
    val_end = min(val_end, len(ids))

    return {
        "train": ids[:train_end],
        "val": ids[train_end:val_end],
        "test": ids[val_end:],
    }


def build_windows(
    episodes: Sequence[EpisodeSpec],
    split_by_episode: dict[str, str],
    history: int,
    horizon: int,
    stride: int,
    exclude_cross_boundary: bool = False,
) -> list[WindowRecord]:
    if history <= 0:
        raise ValueError("history must be positive.")
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    records: list[WindowRecord] = []
    for episode in episodes:
        meta = episode.meta
        split = split_by_episode.get(meta.episode_id)
        if split is None:
            raise ValueError(f"Missing split for episode {meta.episode_id}.")

        start_t = history - 1
        end_t = meta.num_frames - horizon
        for t in range(start_t, end_t + 1, stride):
            try:
                if episode.potential is not None:
                    label = compute_window_label_from_potential(
                        episode.boundaries,
                        episode.potential,
                        t=t,
                        horizon=horizon,
                    )
                else:
                    label = compute_window_label(episode.boundaries, t=t, horizon=horizon)
            except ValueError:
                continue
            if exclude_cross_boundary and label.cross_boundary:
                continue

            history_indices = tuple(range(t - history + 1, t + 1))
            future_indices = tuple(range(t, t + horizon))
            records.append(
                WindowRecord(
                    window_id=f"{meta.episode_id}_t{t:06d}",
                    episode_id=meta.episode_id,
                    task_id=meta.task_id,
                    t=t,
                    history_indices=history_indices,
                    future_indices=future_indices,
                    stage=label.stage,
                    stage_id=label.stage_id,
                    split=split,
                    cross_boundary=label.cross_boundary,
                    primitive_time=label.primitive_time,
                    delta_phi=label.delta_phi,
                    is_success=meta.success,
                    phi_t=label.phi_t,
                    phi_future=label.phi_future,
                    delta_phi_raw=label.delta_phi_raw,
                    source=meta.source,
                )
            )
    return records


def episode_to_split(split: dict[str, Sequence[str]]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for split_name, episode_ids in split.items():
        for episode_id in episode_ids:
            if episode_id in mapping:
                raise ValueError(f"Episode {episode_id} appears in multiple splits.")
            mapping[episode_id] = split_name
    return mapping
