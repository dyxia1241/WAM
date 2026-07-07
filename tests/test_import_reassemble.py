from __future__ import annotations

import numpy as np

from ppwam.import_reassemble import build_primitive_boundaries, prompt_records_for_recordings, recording_language


def _recording_row() -> dict:
    return {
        "recording_id": "rec_001",
        "split": "test_split1",
        "high_level_texts": ["No action", "Pick USB", "Pick USB", "Insert USB"],
        "segments": [
            {
                "segment_index": 0,
                "text": "No action",
                "success": True,
                "start": 0.0,
                "end": 5.0,
                "low_level": [],
            },
            {
                "segment_index": 1,
                "text": "Pick USB",
                "success": True,
                "start": 10.0,
                "end": 40.0,
                "low_level": [
                    {"low_index": 0, "text": "Approach", "success": True, "start": 10.0, "end": 18.0},
                    {"low_index": 1, "text": "Grasp", "success": True, "start": 18.0, "end": 28.0},
                    {"low_index": 2, "text": "Lift", "success": True, "start": 28.0, "end": 40.0},
                ],
            },
            {
                "segment_index": 2,
                "text": "Insert USB",
                "success": False,
                "start": 45.0,
                "end": 60.0,
                "low_level": [
                    {"low_index": 0, "text": "Push", "success": True, "start": 45.0, "end": 60.0},
                ],
            },
        ],
    }


def test_reassemble_boundaries_use_successful_low_level_segments() -> None:
    timestamps = np.arange(0, 80, dtype=np.float64)

    boundaries, metadata = build_primitive_boundaries(_recording_row(), timestamps=timestamps, min_stage_span=4)

    assert [item["stage"] for item in boundaries] == ["approach", "grasp", "move"]
    assert all(item["start"] <= item["end"] for item in boundaries)
    assert {item["high_level_text_raw"] for item in metadata} == {"Pick USB"}


def test_reassemble_prompt_records_use_recording_id_and_action_chain() -> None:
    records = prompt_records_for_recordings([_recording_row()])

    assert len(records) == 1
    assert records[0].task_id == "rec_001"
    assert records[0].primitive_chain == ("Pick USB", "Insert USB")
    assert "Pick USB" in records[0].prompt


def test_reassemble_recording_language_drops_no_action_and_consecutive_duplicates() -> None:
    assert recording_language(_recording_row()) == "Pick USB; Insert USB"
