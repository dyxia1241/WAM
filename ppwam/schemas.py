from __future__ import annotations

from dataclasses import dataclass

from ppwam.labels import PrimitiveBoundary, WindowLabel


@dataclass(frozen=True)
class EpisodeMeta:
    episode_id: str
    task_id: str
    num_frames: int
    action_dim: int
    proprio_dim: int
    source: str = ""
    cameras: tuple[str, ...] = ("cam0",)
    fps: int = 10
    language: str = ""
    success: bool = True


@dataclass(frozen=True)
class EpisodeSpec:
    meta: EpisodeMeta
    boundaries: tuple[PrimitiveBoundary, ...]
    potential: tuple[float, ...] | None = None


@dataclass(frozen=True)
class WindowRecord:
    window_id: str
    episode_id: str
    task_id: str
    t: int
    history_indices: tuple[int, ...]
    future_indices: tuple[int, ...]
    stage: str
    stage_id: int
    split: str
    cross_boundary: bool
    primitive_time: float
    delta_phi: float
    is_success: bool
    phi_t: float = 0.0
    phi_future: float = 0.0
    delta_phi_raw: float = 0.0
    source: str = ""
    source_id: int = -1


__all__ = [
    "EpisodeMeta",
    "EpisodeSpec",
    "PrimitiveBoundary",
    "WindowLabel",
    "WindowRecord",
]
