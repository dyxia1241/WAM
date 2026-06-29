from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from mvp0 import import_gm100


class FakeColumn:
    def __init__(self, values):
        self._values = values

    def to_pylist(self):
        return self._values


class FakeTable:
    def __init__(self, columns):
        self._columns = columns
        self.column_names = list(columns)

    def __getitem__(self, key):
        return FakeColumn(self._columns[key])


def test_manifest_selection_and_episode_id() -> None:
    manifest = {
        "selected_episodes": [
            {"task_id": "task_00001", "episode_id": "episode_000006"},
            {"task_id": "task_00002", "episode_id": "episode_000041"},
        ]
    }

    selected = import_gm100.selected_episodes_from_manifest(manifest)

    assert selected == [
        import_gm100.GM100EpisodeSelection("task_00001", "episode_000006"),
        import_gm100.GM100EpisodeSelection("task_00002", "episode_000041"),
    ]
    assert import_gm100.wam_episode_id("task_00001", "episode_000006") == "task_00001_episode_000006"


def test_camera_feature_mapping() -> None:
    info = {
        "features": {
            "observation.images.camera_wrist_left": {"dtype": "video"},
            "observation.state.arm.position": {"dtype": "float32"},
            "observation.images.camera_top": {"dtype": "video"},
        }
    }

    assert import_gm100.camera_feature_to_name("observation.images.camera_top") == "camera_top"
    assert import_gm100.camera_name_to_feature("camera_wrist_left") == "observation.images.camera_wrist_left"
    assert import_gm100.camera_features_from_info(info) == [
        "observation.images.camera_top",
        "observation.images.camera_wrist_left",
    ]


def test_episode_language_and_length_from_meta(tmp_path) -> None:
    task_root = tmp_path / "task_00001" / "meta"
    task_root.mkdir(parents=True)
    (task_root / "episodes.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"episode_index": 6, "tasks": ["hit-ball"], "length": 428}),
                json.dumps({"episode_index": 98, "tasks": ["hit-ball"], "length": 512}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (task_root / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "fallback"}) + "\n")

    assert import_gm100.task_language(tmp_path, "task_00001", "episode_000006") == "hit-ball"
    assert import_gm100.episode_length(tmp_path, "task_00001", "episode_000098") == 512


def test_column_to_array_and_slice_arrays() -> None:
    table = FakeTable(
        {
            "vector": [[1.0, 2.0], [3.0, 4.0]],
            "scalar": [5, 6],
        }
    )

    vector = import_gm100.column_to_array(table, "vector")
    scalar = import_gm100.column_to_array(table, "scalar", dtype=np.int64)
    sliced = import_gm100.slice_arrays({"vector": vector, "scalar": scalar.squeeze(axis=1)}, 1)

    assert vector.shape == (2, 2)
    assert scalar.shape == (2, 1)
    assert sliced["vector"].shape == (1, 2)
    assert sliced["scalar"].shape == (1,)


def test_import_one_episode_can_skip_image_extraction(tmp_path, monkeypatch) -> None:
    raw_root = tmp_path / "raw"
    meta_root = raw_root / "task_00001" / "meta"
    meta_root.mkdir(parents=True)
    (meta_root / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "features": {
                    "observation.images.camera_top": {"dtype": "video"},
                },
            }
        ),
        encoding="utf-8",
    )
    (meta_root / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "tasks": ["pick"], "length": 4}) + "\n",
        encoding="utf-8",
    )
    (meta_root / "tasks.jsonl").write_text(json.dumps({"task": "fallback"}) + "\n", encoding="utf-8")

    def fake_read_parquet_arrays(_path):
        return {
            "proprio": np.zeros((4, 3), dtype=np.float32),
            "action": np.zeros((4, 2), dtype=np.float32),
        }

    def fail_extract_video_frames(*_args, **_kwargs):
        raise AssertionError("video extraction should be skipped")

    monkeypatch.setattr(import_gm100, "read_parquet_arrays", fake_read_parquet_arrays)
    monkeypatch.setattr(import_gm100, "extract_video_frames", fail_extract_video_frames)

    imported = import_gm100.import_one_episode(
        raw_root=raw_root,
        output_root=tmp_path / "episodes",
        selection=import_gm100.GM100EpisodeSelection("task_00001", "episode_000000"),
        jpeg_quality=95,
        max_frames=None,
        overwrite=False,
        skip_images=True,
    )

    episode_dir = Path(imported.output_dir)
    meta = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
    import_manifest = json.loads((episode_dir / "import_manifest.json").read_text(encoding="utf-8"))
    assert meta["images_imported"] is False
    assert import_manifest["images_imported"] is False
    assert not (episode_dir / "images").exists()
