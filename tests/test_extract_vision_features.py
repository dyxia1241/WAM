from __future__ import annotations

import json

import numpy as np

from ppwam.extract_vision_features import episode_records, extract_mock_features, gm100_video_path
from ppwam.features import read_feature_store


def write_episode(root, episode_id: str, source_episode_id: str, labeled: bool) -> None:
    episode_dir = root / episode_id
    episode_dir.mkdir(parents=True)
    (episode_dir / "meta.json").write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "task_id": "task_00001",
                "source_episode_id": source_episode_id,
                "num_frames": 3,
                "cameras": ["camera_top"],
                "action_dim": 2,
                "proprio_dim": 2,
            }
        ),
        encoding="utf-8",
    )
    if labeled:
        (episode_dir / "labels.json").write_text(
            json.dumps({"primitive_boundaries": [{"stage": "move", "start": 0, "end": 2}]}),
            encoding="utf-8",
        )


def test_episode_records_can_skip_missing_labels(tmp_path) -> None:
    write_episode(tmp_path, "task_00001_episode_000000", "episode_000000", labeled=True)
    write_episode(tmp_path, "task_00001_episode_000001", "episode_000001", labeled=False)

    records = episode_records(tmp_path, skip_missing_labels=True)

    assert [record[1].episode_id for record in records] == ["task_00001_episode_000000"]


def test_mock_features_honor_label_filter_and_limit(tmp_path) -> None:
    episodes = tmp_path / "episodes"
    write_episode(episodes, "task_00001_episode_000000", "episode_000000", labeled=True)
    write_episode(episodes, "task_00001_episode_000001", "episode_000001", labeled=True)
    write_episode(episodes, "task_00001_episode_000002", "episode_000002", labeled=False)

    count = extract_mock_features(
        episodes=episodes,
        output=tmp_path / "features",
        feature_dim=5,
        seed=0,
        skip_missing_labels=True,
        limit_episodes=1,
    )

    assert count == 1
    loaded = read_feature_store(tmp_path / "features" / "task_00001_episode_000000.npz")
    assert loaded["camera_top"].shape == (3, 5)
    assert loaded["camera_top"].dtype == np.float16
    assert not (tmp_path / "features" / "task_00001_episode_000001.npz").exists()


def test_gm100_video_path_uses_camera_feature_layout() -> None:
    path = gm100_video_path("/raw", "task_00001", "episode_000123", "camera_top")

    assert str(path) == "/raw/task_00001/videos/chunk-000/observation.images.camera_top/episode_000123.mp4"
