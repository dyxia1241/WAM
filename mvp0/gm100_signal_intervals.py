from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


EFFORT_COLUMN = "observation.state.effector.effort"
VELOCITY_COLUMN = "observation.state.arm.velocity"
FRAME_INDEX_COLUMN = "frame_index"

IMMEDIATE_GAP_MAX = 15
SHORT_GAP_MAX = 45
MEDIUM_GAP_MAX = 90
ANCHOR_OVERLAP_MERGE_GAP = 3

SERIAL_REPEAT_PATTERNS = (
    r"one[- ]by[- ]one",
    r"\bitems?\b",
    r"\bobjects?\b",
    r"\bmultiple\b",
    r"\bseveral\b",
    r"\bsort\b",
    r"\borganize\b",
    r"\bstack\b",
    r"\barrange\b",
    r"\bbus-table\b",
    r"\btable-setting\b",
    r"\bput-coins\b",
    r"\bremove-objects\b",
    r"\bplace-seven\b",
)


@dataclass(frozen=True)
class EpisodeSignals:
    effort: np.ndarray
    velocity: np.ndarray
    frame_index: np.ndarray

    @property
    def num_rows(self) -> int:
        return int(self.effort.shape[0])


@dataclass(frozen=True)
class RawEvent:
    raw_event_id: str
    arm: str
    contact_row: int
    release_row: int


@dataclass(frozen=True)
class AnchorEvent:
    anchor_event_id: str
    anchor_order: int
    anchor_start_row: int
    anchor_end_row: int
    active_arm_pattern: str
    source_raw_event_ids: tuple[str, ...]
    source_arms: tuple[str, ...]


@dataclass(frozen=True)
class LocalStepInterval:
    interval_id: str
    interval_order: int
    start_row: int
    end_row: int
    active_arm_pattern: str
    source_anchor_event_ids: tuple[str, ...]
    source_raw_event_ids: tuple[str, ...]
    merge_confidence: str
    reason_codes: tuple[str, ...]
    serial_repetition_risk: str

    @property
    def span(self) -> int:
        return int(self.end_row - self.start_row)


@dataclass(frozen=True)
class WAMBoundaryCandidate:
    stage: str
    start: int
    end: int
    interval: LocalStepInterval
    feasible_windows: int


def dataclass_json(item: object) -> dict[str, Any]:
    payload = asdict(item)
    for key, value in list(payload.items()):
        if isinstance(value, tuple):
            payload[key] = list(value)
    return payload


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def load_task_annotations(annotation_csv: str | Path) -> dict[str, dict[str, Any]]:
    annotation_csv = Path(annotation_csv)
    with annotation_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        if not task_id:
            continue
        normalized = dict(row)
        for key in ("has_gripper_motion", "t2_eligible", "t4_eligible"):
            if key in normalized:
                normalized[key] = parse_bool(normalized[key])
        out[task_id] = normalized
    return out


def load_task_name(raw_root: str | Path, task_id: str) -> str:
    tasks_path = Path(raw_root) / task_id / "meta" / "tasks.jsonl"
    if not tasks_path.exists():
        return ""
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            loaded = json.loads(line)
            if isinstance(loaded, dict):
                return str(loaded.get("task", ""))
    return ""


def require_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required to read GM-100 parquet files.") from exc
    return pq


def _column_to_2d_array(table: Any, column: str, dtype: np.dtype | type) -> np.ndarray:
    if column not in table.column_names:
        raise ValueError(f"Missing parquet column: {column}")
    values = table[column].to_pylist()
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Column {column} must convert to a 2D array; got {array.shape}.")
    return array


def read_episode_signals(parquet_path: str | Path) -> EpisodeSignals:
    parquet_path = Path(parquet_path)
    pq = require_pyarrow_parquet()
    schema = pq.read_schema(parquet_path)
    columns = [EFFORT_COLUMN, VELOCITY_COLUMN]
    has_frame_index = FRAME_INDEX_COLUMN in schema.names
    if has_frame_index:
        columns.append(FRAME_INDEX_COLUMN)
    table = pq.read_table(parquet_path, columns=columns)
    effort = _column_to_2d_array(table, EFFORT_COLUMN, np.float32)
    velocity = _column_to_2d_array(table, VELOCITY_COLUMN, np.float32)
    if has_frame_index:
        frame_index = _column_to_2d_array(table, FRAME_INDEX_COLUMN, np.int64).squeeze(axis=1)
    else:
        frame_index = np.arange(effort.shape[0], dtype=np.int64)
    if effort.shape[0] != velocity.shape[0] or effort.shape[0] != frame_index.shape[0]:
        raise ValueError(f"Signal frame count mismatch in {parquet_path}.")
    if effort.shape[1] < 2:
        raise ValueError(f"{EFFORT_COLUMN} must contain at least left/right effort values.")
    return EpisodeSignals(effort=effort[:, :2], velocity=velocity, frame_index=frame_index)


def estimate_baseline(effort: np.ndarray, tail_ratio: float = 0.10, head_ratio: float = 0.10) -> dict[str, float | int]:
    if effort.ndim != 2 or effort.shape[1] < 2:
        raise ValueError("effort must be a 2D array with at least two columns.")
    n = len(effort)
    if n == 0:
        raise ValueError("Empty effort array.")

    tail_n = max(1, int(n * tail_ratio))
    head_n = max(1, int(n * head_ratio))
    tail = effort[-tail_n:]
    head = effort[:head_n]
    return {
        "baseline_left": float(np.median(tail[:, 0])),
        "baseline_right": float(np.median(tail[:, 1])),
        "sigma_left": float(np.std(head[:, 0], ddof=0)),
        "sigma_right": float(np.std(head[:, 1], ddof=0)),
        "tail_frames": tail_n,
        "head_frames": head_n,
    }


def _detect_single_arm_events(
    effort: np.ndarray,
    baseline: float,
    sigma: float,
    min_persist_frames: int = 5,
    contact_sigma_k: float = 3.0,
    release_sigma_k: float = 2.0,
) -> list[dict[str, int]]:
    sigma_eff = max(float(sigma), 1e-6)
    contact_th = float(baseline) - contact_sigma_k * sigma_eff
    release_lo = float(baseline) - release_sigma_k * sigma_eff
    release_hi = float(baseline) + release_sigma_k * sigma_eff

    events: list[dict[str, int]] = []
    state = "idle"
    contact_row: int | None = None
    i = 0
    n = len(effort)
    while i < n:
        value = float(effort[i])
        if state == "idle":
            j = min(n, i + min_persist_frames)
            if j - i == min_persist_frames and np.all(effort[i:j] < contact_th):
                contact_row = i
                state = "contacting"
                i = j
                continue
        elif release_lo <= value <= release_hi:
            events.append({"contact_frame": int(contact_row), "release_frame": int(i)})
            contact_row = None
            state = "idle"
        i += 1
    return events


def detect_contact_events(effort: np.ndarray, task_meta: dict[str, Any], min_persist_frames: int = 5) -> dict[str, list[dict[str, int]]]:
    if not parse_bool(task_meta.get("has_gripper_motion", False)):
        return {"left": [], "right": []}

    stats = estimate_baseline(effort)
    arm_type = str(task_meta.get("arm_type", "unknown"))
    out: dict[str, list[dict[str, int]]] = {"left": [], "right": []}
    if arm_type in {"single_left", "bimanual_sync", "bimanual_sequential"}:
        out["left"] = _detect_single_arm_events(
            effort[:, 0],
            baseline=float(stats["baseline_left"]),
            sigma=float(stats["sigma_left"]),
            min_persist_frames=min_persist_frames,
        )
    if arm_type in {"single_right", "bimanual_sync", "bimanual_sequential"}:
        out["right"] = _detect_single_arm_events(
            effort[:, 1],
            baseline=float(stats["baseline_right"]),
            sigma=float(stats["sigma_right"]),
            min_persist_frames=min_persist_frames,
        )
    return out


def _select_velocity_signal(velocity: np.ndarray, task_meta: dict[str, Any]) -> np.ndarray:
    if velocity.ndim != 2 or velocity.shape[1] < 2:
        raise ValueError("velocity must be a 2D array with at least two columns.")
    half = velocity.shape[1] // 2
    vel_l = np.linalg.norm(velocity[:, :half], axis=1)
    vel_r = np.linalg.norm(velocity[:, half:], axis=1)
    primary_arm = str(task_meta.get("primary_arm", "none"))
    if primary_arm == "left":
        return vel_l
    if primary_arm == "right":
        return vel_r
    return np.maximum(vel_l, vel_r)


def infer_serial_repetition_risk(task_name_raw: str) -> str:
    text = str(task_name_raw or "").strip().lower()
    if not text:
        return "unknown"
    score = 0
    for pattern in SERIAL_REPEAT_PATTERNS:
        if re.search(pattern, text):
            score += 1
    if re.search(r"\b(seven|eight|nine|ten|twelve)\b", text):
        score += 1
    if score >= 2:
        return "high"
    if score == 1:
        return "medium"
    return "low"


def build_raw_events(task_id: str, episode_id: int, contact_events: dict[str, list[dict[str, int]]]) -> list[RawEvent]:
    out: list[RawEvent] = []
    event_idx = 1
    for arm in ("left", "right"):
        for event in contact_events.get(arm, []):
            contact_row = int(event["contact_frame"])
            release_row = int(event["release_frame"])
            if release_row <= contact_row:
                continue
            out.append(
                RawEvent(
                    raw_event_id=f"{task_id}__{episode_id}__re{event_idx:03d}",
                    arm=arm,
                    contact_row=contact_row,
                    release_row=release_row,
                )
            )
            event_idx += 1
    out.sort(key=lambda item: (item.contact_row, item.release_row, item.arm))
    return out


def _active_arm_pattern(arms: set[str]) -> str:
    if arms == {"left"}:
        return "left"
    if arms == {"right"}:
        return "right"
    if arms == {"left", "right"}:
        return "both"
    return "unknown"


def build_anchor_events(task_id: str, episode_id: int, raw_events: list[RawEvent]) -> list[AnchorEvent]:
    if not raw_events:
        return []
    anchors: list[AnchorEvent] = []
    current_group = [raw_events[0]]
    current_end = raw_events[0].release_row

    def flush(group_events: list[RawEvent], order: int) -> AnchorEvent:
        arms = {event.arm for event in group_events}
        return AnchorEvent(
            anchor_event_id=f"{task_id}__{episode_id}__a{order:03d}",
            anchor_order=order,
            anchor_start_row=min(event.contact_row for event in group_events),
            anchor_end_row=max(event.release_row for event in group_events),
            active_arm_pattern=_active_arm_pattern(arms),
            source_raw_event_ids=tuple(event.raw_event_id for event in group_events),
            source_arms=tuple(sorted(arms)),
        )

    for raw_event in raw_events[1:]:
        if raw_event.contact_row <= current_end + ANCHOR_OVERLAP_MERGE_GAP:
            current_group.append(raw_event)
            current_end = max(current_end, raw_event.release_row)
            continue
        anchors.append(flush(current_group, len(anchors) + 1))
        current_group = [raw_event]
        current_end = raw_event.release_row
    anchors.append(flush(current_group, len(anchors) + 1))
    return anchors


def gap_bucket(gap_rows: int) -> str:
    if gap_rows <= IMMEDIATE_GAP_MAX:
        return "immediate"
    if gap_rows <= SHORT_GAP_MAX:
        return "short"
    if gap_rows <= MEDIUM_GAP_MAX:
        return "medium"
    return "long"


def transition_strength(velocity_signal: np.ndarray, prev_end: int, next_start: int) -> str:
    if next_start <= prev_end:
        return "low"
    seg = velocity_signal[prev_end:next_start]
    if len(seg) == 0:
        return "low"
    peak = float(np.max(seg))
    mean = float(np.mean(seg))
    if peak < 0.45 and mean < 0.25:
        return "low"
    if peak < 1.00 and mean < 0.55:
        return "medium"
    return "high"


def pairwise_features(
    prev_anchor: AnchorEvent,
    next_anchor: AnchorEvent,
    velocity_signal: np.ndarray,
    task_serial_risk: str,
) -> dict[str, Any]:
    gap_rows = int(next_anchor.anchor_start_row - prev_anchor.anchor_end_row)
    same_active_arm_pattern = "yes" if prev_anchor.active_arm_pattern == next_anchor.active_arm_pattern else "no"
    inter_anchor_transition_strength = transition_strength(velocity_signal, prev_anchor.anchor_end_row, next_anchor.anchor_start_row)

    if gap_rows > SHORT_GAP_MAX:
        inter_anchor_reset_hint = "yes"
    elif gap_rows <= IMMEDIATE_GAP_MAX:
        inter_anchor_reset_hint = "no"
    else:
        inter_anchor_reset_hint = "unknown"

    explicit_retry_hint = "unknown"
    if gap_rows <= IMMEDIATE_GAP_MAX and same_active_arm_pattern == "yes" and inter_anchor_transition_strength == "low":
        explicit_retry_hint = "yes"
    elif gap_rows > SHORT_GAP_MAX or same_active_arm_pattern == "no":
        explicit_retry_hint = "no"

    serial_repetition_hint = "no"
    if task_serial_risk == "high" and gap_rows > 8:
        serial_repetition_hint = "yes"
    elif task_serial_risk == "medium" and gap_rows > IMMEDIATE_GAP_MAX:
        serial_repetition_hint = "yes"

    return {
        "gap_rows": gap_rows,
        "gap_bucket": gap_bucket(gap_rows),
        "same_active_arm_pattern": same_active_arm_pattern,
        "inter_anchor_reset_hint": inter_anchor_reset_hint,
        "explicit_retry_hint": explicit_retry_hint,
        "serial_repetition_hint": serial_repetition_hint,
        "inter_anchor_transition_strength": inter_anchor_transition_strength,
    }


def should_merge_pair(features: dict[str, Any], current_interval_anchor_count: int) -> tuple[bool, str, list[str]]:
    reason_codes: list[str] = []
    if features["gap_bucket"] == "long":
        return False, "low", ["long_gap_block"]
    if features["serial_repetition_hint"] == "yes":
        return False, "low", ["serial_repetition_risk_block"]
    if features["same_active_arm_pattern"] == "no":
        return False, "low", ["arm_pattern_mismatch_block"]
    if features["inter_anchor_reset_hint"] == "yes" and features["explicit_retry_hint"] != "yes":
        return False, "low", ["reset_hint_block"]

    if features["gap_bucket"] == "immediate":
        reason_codes.append("adjacent_immediate_gap")
    elif features["gap_bucket"] == "short":
        reason_codes.append("adjacent_short_gap")
    if features["same_active_arm_pattern"] == "yes":
        reason_codes.append("same_arm_pattern")
    if features["explicit_retry_hint"] == "yes":
        reason_codes.append("explicit_retry_hint")
    if features["inter_anchor_transition_strength"] == "low":
        reason_codes.append("low_transition_strength")
    if features["inter_anchor_reset_hint"] == "no":
        reason_codes.append("no_reset_evidence")

    if (
        features["gap_bucket"] in {"immediate", "short"}
        and features["explicit_retry_hint"] == "yes"
        and features["same_active_arm_pattern"] != "no"
        and features["inter_anchor_reset_hint"] == "no"
        and features["serial_repetition_hint"] != "yes"
        and current_interval_anchor_count + 1 <= 3
    ):
        return True, "high", reason_codes

    if (
        features["gap_bucket"] == "immediate"
        and features["explicit_retry_hint"] != "no"
        and features["same_active_arm_pattern"] == "yes"
        and features["inter_anchor_reset_hint"] == "no"
        and features["serial_repetition_hint"] == "no"
        and features["inter_anchor_transition_strength"] == "low"
        and current_interval_anchor_count + 1 <= 2
    ):
        return True, "medium", reason_codes

    return False, "low", reason_codes or ["uncertain_pair_keep_split"]


def build_local_step_intervals_for_episode(
    task_id: str,
    episode_id: int,
    signals: EpisodeSignals,
    task_meta: dict[str, Any],
    task_name_raw: str = "",
) -> list[LocalStepInterval]:
    if signals.num_rows == 0 or not parse_bool(task_meta.get("has_gripper_motion", False)):
        return []
    contact_events = detect_contact_events(signals.effort, task_meta=task_meta, min_persist_frames=5)
    raw_events = build_raw_events(task_id, episode_id, contact_events)
    anchors = build_anchor_events(task_id, episode_id, raw_events)
    if not anchors:
        return []

    velocity_signal = _select_velocity_signal(signals.velocity, task_meta)
    serial_risk = infer_serial_repetition_risk(task_name_raw)
    intervals: list[LocalStepInterval] = []
    current: dict[str, Any] = {
        "interval_id": f"{task_id}__{episode_id}__lsi001",
        "interval_order": 1,
        "start_row": anchors[0].anchor_start_row,
        "end_row": anchors[0].anchor_end_row,
        "active_arm_patterns": [anchors[0].active_arm_pattern],
        "source_anchor_event_ids": [anchors[0].anchor_event_id],
        "source_raw_event_ids": list(anchors[0].source_raw_event_ids),
        "merge_confidence": "low",
        "reason_codes": [],
    }
    last_anchor = anchors[0]

    def flush_interval(record: dict[str, Any]) -> LocalStepInterval:
        patterns = list(dict.fromkeys(record["active_arm_patterns"]))
        if patterns == ["left"]:
            active_arm_pattern = "left"
        elif patterns == ["right"]:
            active_arm_pattern = "right"
        elif set(patterns) == {"left", "right", "both"} or set(patterns) == {"left", "right"} or patterns == ["both"]:
            active_arm_pattern = "both"
        else:
            active_arm_pattern = patterns[0] if patterns else "unknown"
        return LocalStepInterval(
            interval_id=str(record["interval_id"]),
            interval_order=int(record["interval_order"]),
            start_row=int(record["start_row"]),
            end_row=int(record["end_row"]),
            active_arm_pattern=active_arm_pattern,
            source_anchor_event_ids=tuple(record["source_anchor_event_ids"]),
            source_raw_event_ids=tuple(record["source_raw_event_ids"]),
            merge_confidence=str(record["merge_confidence"]),
            reason_codes=tuple(sorted(set(str(code) for code in record["reason_codes"]))),
            serial_repetition_risk=serial_risk,
        )

    for next_anchor in anchors[1:]:
        features = pairwise_features(last_anchor, next_anchor, velocity_signal, serial_risk)
        merge, confidence, reason_codes = should_merge_pair(features, current_interval_anchor_count=len(current["source_anchor_event_ids"]))
        if merge:
            current["source_anchor_event_ids"].append(next_anchor.anchor_event_id)
            current["source_raw_event_ids"].extend(next_anchor.source_raw_event_ids)
            current["end_row"] = next_anchor.anchor_end_row
            current["active_arm_patterns"].append(next_anchor.active_arm_pattern)
            current["reason_codes"].extend(reason_codes)
            if confidence == "high":
                current["merge_confidence"] = "high"
            elif current["merge_confidence"] != "high":
                current["merge_confidence"] = "medium"
        else:
            intervals.append(flush_interval(current))
            current = {
                "interval_id": f"{task_id}__{episode_id}__lsi{len(intervals) + 1:03d}",
                "interval_order": len(intervals) + 1,
                "start_row": next_anchor.anchor_start_row,
                "end_row": next_anchor.anchor_end_row,
                "active_arm_patterns": [next_anchor.active_arm_pattern],
                "source_anchor_event_ids": [next_anchor.anchor_event_id],
                "source_raw_event_ids": list(next_anchor.source_raw_event_ids),
                "merge_confidence": "low",
                "reason_codes": [],
            }
        last_anchor = next_anchor

    intervals.append(flush_interval(current))
    return intervals


def count_feasible_windows(
    start: int,
    end: int,
    num_frames: int,
    history: int,
    horizon: int,
    stride: int,
    exclude_cross_boundary: bool = True,
) -> int:
    if history <= 0 or horizon <= 0 or stride <= 0:
        raise ValueError("history, horizon, and stride must be positive.")
    if end < start:
        return 0

    grid_start = history - 1
    lo = max(grid_start, int(start))
    hi = min(int(num_frames) - horizon, int(end))
    if exclude_cross_boundary:
        hi = min(hi, int(end) - horizon)
    if hi < lo:
        return 0

    offset = (lo - grid_start) % stride
    first_t = lo if offset == 0 else lo + (stride - offset)
    if first_t > hi:
        return 0
    return int((hi - first_t) // stride + 1)


def interval_to_wam_boundary(
    interval: LocalStepInterval,
    num_frames: int,
    stage: str,
    history: int,
    horizon: int,
    stride: int,
    exclude_cross_boundary: bool = True,
) -> WAMBoundaryCandidate | None:
    start = max(0, int(interval.start_row))
    end = min(int(num_frames) - 1, int(interval.end_row) - 1)
    if end < start:
        return None
    feasible_windows = count_feasible_windows(
        start=start,
        end=end,
        num_frames=num_frames,
        history=history,
        horizon=horizon,
        stride=stride,
        exclude_cross_boundary=exclude_cross_boundary,
    )
    return WAMBoundaryCandidate(stage=stage, start=start, end=end, interval=interval, feasible_windows=feasible_windows)
