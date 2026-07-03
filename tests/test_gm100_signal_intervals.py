from __future__ import annotations

import numpy as np

from ppwam.gm100_signal_intervals import (
    EpisodeSignals,
    RawEvent,
    build_anchor_events,
    build_local_step_intervals_for_episode,
    count_feasible_windows,
    detect_contact_events,
    interval_to_wam_boundary,
)


def test_detect_contact_events_respects_gripper_motion_flag() -> None:
    effort = np.zeros((30, 2), dtype=np.float32)
    effort[5:10, 0] = -1.0

    disabled = detect_contact_events(effort, {"arm_type": "single_left", "has_gripper_motion": False})
    enabled = detect_contact_events(effort, {"arm_type": "single_left", "has_gripper_motion": True})

    assert disabled == {"left": [], "right": []}
    assert enabled["left"] == [{"contact_frame": 5, "release_frame": 10}]
    assert enabled["right"] == []


def test_anchor_events_merge_overlapping_arm_contacts() -> None:
    raw_events = [
        RawEvent("e1", "left", 5, 12),
        RawEvent("e2", "right", 9, 14),
    ]

    anchors = build_anchor_events("task_00001", 0, raw_events)

    assert len(anchors) == 1
    assert anchors[0].anchor_start_row == 5
    assert anchors[0].anchor_end_row == 14
    assert anchors[0].active_arm_pattern == "both"


def test_local_step_intervals_merge_immediate_retry_like_contacts() -> None:
    effort = np.zeros((40, 2), dtype=np.float32)
    effort[5:10, 0] = -1.0
    effort[14:19, 0] = -1.0
    velocity = np.zeros((40, 12), dtype=np.float32)
    signals = EpisodeSignals(effort=effort, velocity=velocity, frame_index=np.arange(40))

    intervals = build_local_step_intervals_for_episode(
        task_id="task_00001",
        episode_id=0,
        signals=signals,
        task_meta={"arm_type": "single_left", "primary_arm": "left", "has_gripper_motion": True},
        task_name_raw="pick up the object",
    )

    assert len(intervals) == 1
    assert intervals[0].start_row == 5
    assert intervals[0].end_row == 19
    assert intervals[0].merge_confidence == "high"


def test_count_feasible_windows_matches_prepare_windows_grid() -> None:
    assert count_feasible_windows(start=5, end=29, num_frames=50, history=4, horizon=8, stride=2) == 9
    assert count_feasible_windows(start=5, end=10, num_frames=50, history=4, horizon=8, stride=2) == 0


def test_interval_to_wam_boundary_converts_half_open_end_to_inclusive() -> None:
    effort = np.zeros((50, 2), dtype=np.float32)
    effort[10:35, 0] = -1.0
    velocity = np.zeros((50, 12), dtype=np.float32)
    signals = EpisodeSignals(effort=effort, velocity=velocity, frame_index=np.arange(50))
    interval = build_local_step_intervals_for_episode(
        task_id="task_00001",
        episode_id=0,
        signals=signals,
        task_meta={"arm_type": "single_left", "primary_arm": "left", "has_gripper_motion": True},
        task_name_raw="pick up the object",
    )[0]

    candidate = interval_to_wam_boundary(interval, num_frames=50, stage="move", history=4, horizon=8, stride=2)

    assert candidate is not None
    assert candidate.start == 10
    assert candidate.end == 34
    assert candidate.feasible_windows > 0
