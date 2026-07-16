from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


STAGE_VOCAB = ("approach", "grasp", "move", "place", "release")
STAGE_TO_ID = {stage: idx for idx, stage in enumerate(STAGE_VOCAB)}


@dataclass(frozen=True)
class PrimitiveBoundary:
    stage: str
    start: int
    end: int


@dataclass(frozen=True)
class WindowLabel:
    delta_phi: float
    primitive_time: float
    phi_t: float
    phi_future: float
    delta_phi_raw: float
    stage: str
    stage_id: int
    cross_boundary: bool


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def validate_boundaries(boundaries: Iterable[PrimitiveBoundary]) -> list[PrimitiveBoundary]:
    checked = list(boundaries)
    if not checked:
        raise ValueError("At least one primitive boundary is required.")

    previous_end = None
    for boundary in checked:
        if boundary.stage not in STAGE_TO_ID:
            raise ValueError(f"Unknown stage: {boundary.stage}")
        if boundary.start < 0 or boundary.end < 0:
            raise ValueError("Boundary start/end must be non-negative.")
        if boundary.end < boundary.start:
            raise ValueError("Boundary end must be >= start.")
        if previous_end is not None and boundary.start < previous_end:
            raise ValueError("Primitive boundaries must be sorted and non-overlapping.")
        previous_end = boundary.end
    return checked


def primitive_phi(t: int, boundary: PrimitiveBoundary) -> float:
    denom = max(boundary.end - boundary.start, 1)
    return clamp((t - boundary.start) / denom)


def find_boundary_index(boundaries: Iterable[PrimitiveBoundary], t: int) -> int:
    checked = validate_boundaries(boundaries)
    if t <= checked[0].start:
        return 0
    for index, boundary in enumerate(checked):
        if boundary.start <= t <= boundary.end:
            return index
        if index + 1 < len(checked) and boundary.end < t < checked[index + 1].start:
            return index
    return len(checked) - 1


def find_boundary(boundaries: Iterable[PrimitiveBoundary], t: int) -> PrimitiveBoundary:
    checked = validate_boundaries(boundaries)
    for boundary in checked:
        if boundary.start <= t <= boundary.end:
            return boundary
    raise ValueError(f"No primitive boundary contains frame t={t}.")


def global_potential(boundaries: Iterable[PrimitiveBoundary], t: int) -> float:
    checked = validate_boundaries(boundaries)
    index = find_boundary_index(checked, t)
    boundary = checked[index]
    primitive_progress = primitive_phi(t, boundary)
    if t > boundary.end:
        primitive_progress = 1.0
    return clamp((float(index) + primitive_progress) / float(len(checked)))


def compute_window_label(
    boundaries: Iterable[PrimitiveBoundary],
    t: int,
    horizon: int,
) -> WindowLabel:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if t < 0:
        raise ValueError("t must be non-negative.")

    checked = validate_boundaries(boundaries)
    boundary = find_boundary(checked, t)
    future_t = t + horizon
    cross_boundary = future_t > boundary.end
    primitive_time = primitive_phi(t, boundary)
    future_phi = primitive_phi(min(future_t, boundary.end), boundary)
    phi_t = global_potential(checked, t)
    phi_future = global_potential(checked, min(future_t, checked[-1].end))

    return WindowLabel(
        delta_phi=max(0.0, future_phi - primitive_time),
        primitive_time=primitive_time,
        phi_t=phi_t,
        phi_future=phi_future,
        delta_phi_raw=phi_future - phi_t,
        stage=boundary.stage,
        stage_id=STAGE_TO_ID[boundary.stage],
        cross_boundary=cross_boundary,
    )


def compute_window_label_from_potential(
    boundaries: Iterable[PrimitiveBoundary],
    potential: Iterable[float],
    t: int,
    horizon: int,
) -> WindowLabel:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if t < 0:
        raise ValueError("t must be non-negative.")

    values = [float(item) for item in potential]
    if not values:
        raise ValueError("potential must not be empty.")
    if t >= len(values):
        raise ValueError(f"Frame t={t} is outside potential length={len(values)}.")
    future_t = min(t + horizon, len(values) - 1)
    checked = validate_boundaries(boundaries)
    boundary = find_boundary(checked, t)
    cross_boundary = t + horizon > boundary.end
    primitive_time = primitive_phi(t, boundary)
    phi_t = clamp(values[t])
    phi_future = clamp(values[future_t])
    raw_gain = phi_future - phi_t

    return WindowLabel(
        delta_phi=max(0.0, raw_gain),
        primitive_time=primitive_time,
        phi_t=phi_t,
        phi_future=phi_future,
        delta_phi_raw=raw_gain,
        stage=boundary.stage,
        stage_id=STAGE_TO_ID[boundary.stage],
        cross_boundary=cross_boundary,
    )
