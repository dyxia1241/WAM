from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ppwam.data import PreparedWindowDataset, collate_batch
from ppwam.features import write_feature_store
from ppwam.joint_flow import JointFlowPreparedWindowDataset
from ppwam.merge_prepared_sources import SourceSpec, merge_prepared_sources
from ppwam.prepare_windows import prepare_windows


def _write_episode(
    root: Path,
    source: str,
    episode_id: str,
    task_id: str,
    dim: int,
    frames: int = 24,
    cameras: tuple[str, ...] = ("cam0",),
) -> None:
    episode_dir = root / episode_id
    episode_dir.mkdir(parents=True)
    (episode_dir / "meta.json").write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "task_id": task_id,
                "source": source,
                "language": f"{source} task",
                "fps": 10,
                "num_frames": frames,
                "cameras": list(cameras),
                "action_dim": dim,
                "proprio_dim": dim,
            }
        ),
        encoding="utf-8",
    )
    (episode_dir / "labels.json").write_text(
        json.dumps(
            {
                "primitive_boundaries": [
                    {"stage": "approach", "start": 0, "end": frames // 2 - 1},
                    {"stage": "move", "start": frames // 2, "end": frames - 1},
                ],
                "success": True,
            }
        ),
        encoding="utf-8",
    )
    rng = np.random.default_rng(abs(hash((source, episode_id))) % 2**32)
    np.savez_compressed(
        episode_dir / "arrays.npz",
        proprio=rng.normal(size=(frames, dim)).astype(np.float32),
        action=rng.normal(size=(frames, dim)).astype(np.float32),
    )


def _write_features(
    root: Path,
    episode_ids: list[str],
    frames: int = 24,
    dim: int = 8,
    cameras: tuple[str, ...] = ("cam0",),
) -> None:
    for episode_id in episode_ids:
        write_feature_store(
            root / f"{episode_id}.npz",
            {
                camera: np.full((frames, dim), index + 1, dtype=np.float16)
                for index, camera in enumerate(cameras)
            },
        )


def _prepare_source(tmp_path: Path, source: str, dim: int, cameras: tuple[str, ...] = ("cam0",)) -> SourceSpec:
    episodes = tmp_path / source / "episodes"
    features = tmp_path / source / "features"
    windows = tmp_path / source / "prepared"
    episode_ids = [f"{source}_ep{i}" for i in range(5)]
    for index, episode_id in enumerate(episode_ids):
        _write_episode(
            episodes,
            source=source,
            episode_id=episode_id,
            task_id=f"task{index % 2}",
            dim=dim,
            cameras=cameras,
        )
    _write_features(features, episode_ids, cameras=cameras)
    prepare_windows(
        episodes,
        windows,
        history=4,
        horizon=4,
        stride=2,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        seed=123,
    )
    return SourceSpec(name=source, windows_dir=windows, episodes_dir=episodes, features_dir=features)


def test_merge_prepared_sources_equalizes_windows_and_routes_loader(tmp_path: Path) -> None:
    source_a = _prepare_source(tmp_path, "source_a", dim=4, cameras=("cam0", "cam1"))
    source_b = _prepare_source(tmp_path, "source_b", dim=2)
    output = tmp_path / "merged"

    index = merge_prepared_sources(
        sources=[source_a, source_b],
        output_dir=output,
        seed=7,
        train_cap=4,
        val_cap=2,
        test_cap=2,
    )

    assert index["counts_by_split_source"] == {
        "train": {"source_a": 4, "source_b": 4},
        "val": {"source_a": 2, "source_b": 2},
        "test": {"source_a": 2, "source_b": 2},
    }
    assert index["params"]["canonical_action_dim"] == 4
    assert index["params"]["canonical_num_cameras"] == 2
    assert index["sources"]["source_a"]["num_cameras"] == 2
    assert index["sources"]["source_b"]["num_cameras"] == 1
    windows = [json.loads(line) for line in (output / "windows.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {row["source"] for row in windows} == {"source_a", "source_b"}
    assert all("::" in row["task_id"] for row in windows)

    dataset = PreparedWindowDataset(
        windows_dir=output,
        episodes_dir=tmp_path / "unused_episodes",
        features_dir=tmp_path / "unused_features",
        split="train",
        feature_dim=8,
    )
    source_a_index = next(
        idx for idx, label_index in enumerate(dataset.indices) if dataset.windows[label_index]["source"] == "source_a"
    )
    source_b_index = next(
        idx for idx, label_index in enumerate(dataset.indices) if dataset.windows[label_index]["source"] == "source_b"
    )
    source_a_sample = dataset[source_a_index]
    sample = dataset[source_b_index]
    assert sample["proprio"].shape == (4,)
    assert sample["action_chunk"].shape == (4, 4)
    assert sample["obs_features"].shape == (4, 2, 8)
    assert np.count_nonzero(sample["obs_features"][:, 1, :].numpy()) == 0
    assert sample["source_id"].item() == index["source_to_id"]["source_b"]

    batch = collate_batch([source_a_sample, sample])
    assert batch["obs_features"].shape == (2, 4, 2, 8)

    joint_dataset = JointFlowPreparedWindowDataset(
        windows_dir=output,
        episodes_dir=tmp_path / "unused_episodes",
        features_dir=tmp_path / "unused_features",
        split="train",
        feature_dim=8,
    )
    joint_batch = collate_batch([joint_dataset[source_a_index], joint_dataset[source_b_index]])
    assert joint_batch["obs_features"].shape == (2, 4, 2, 8)
    assert joint_batch["future_obs_features"].shape == (2, 4, 2, 8)
