from __future__ import annotations

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mvp0.build_gm100_progress_gt import build_progress_gt
from mvp0.prepare_windows import read_episode_spec


def write_jsonl(path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_build_progress_gt_writes_wam_labels(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    task_root = raw_root / "task_00001"
    parquet_dir = task_root / "data" / "chunk-000"
    meta_dir = task_root / "meta"
    parquet_dir.mkdir(parents=True)
    meta_dir.mkdir(parents=True)
    write_jsonl(meta_dir / "tasks.jsonl", [{"task": "pick up the object"}])
    manifest = {
        "selected_episodes": [
            {"task_id": "task_00001", "episode_id": "episode_000000"},
        ]
    }
    (raw_root / "gm100_subset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    effort = np.zeros((60, 2), dtype=np.float32)
    effort[10:40, 0] = -1.0
    velocity = np.zeros((60, 12), dtype=np.float32)
    table = pa.table(
        {
            "observation.state.effector.effort": pa.array(effort.tolist()),
            "observation.state.arm.velocity": pa.array(velocity.tolist()),
            "frame_index": pa.array(np.arange(60, dtype=np.int64)),
        }
    )
    pq.write_table(table, parquet_dir / "episode_000000.parquet")

    annotation = tmp_path / "annotation.csv"
    annotation.write_text(
        "task_id,arm_type,primary_arm,effort_signal_quality,has_gripper_motion\n"
        "task_00001,single_left,left,strong,True\n",
        encoding="utf-8",
    )

    episodes_root = tmp_path / "episodes"
    episode_dir = episodes_root / "task_00001_episode_000000"
    episode_dir.mkdir(parents=True)
    (episode_dir / "meta.json").write_text(
        json.dumps(
            {
                "episode_id": "task_00001_episode_000000",
                "task_id": "task_00001",
                "source_episode_id": "episode_000000",
                "num_frames": 60,
                "cameras": ["camera_top"],
                "action_dim": 14,
                "proprio_dim": 14,
                "success": True,
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(episode_dir / "arrays.npz", action=np.zeros((60, 14)), proprio=np.zeros((60, 14)))

    summary = build_progress_gt(
        raw_root=raw_root,
        annotation_csv=annotation,
        episodes_root=episodes_root,
        output_dir=tmp_path / "gt",
        min_interval_span=24,
        overwrite=True,
    )

    assert summary["counts"]["labels_written"] == 1
    spec = read_episode_spec(episode_dir)
    assert len(spec.boundaries) == 1
    assert spec.boundaries[0].stage == "move"
    assert spec.boundaries[0].start == 10
    assert spec.boundaries[0].end == 39
