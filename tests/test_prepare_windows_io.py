import json
import subprocess
import sys

import numpy as np

from mvp0.data import PreparedWindowDataset, collate_batch
from mvp0.features import write_feature_store
from mvp0.prepare_windows import prepare_windows, read_episode_specs
from mvp0.config import apply_overrides, load_config
from mvp0.extract_vision_features import image_paths_for_camera
from mvp0.train import train


def write_toy_episode(root, episode_id: str, task_id: str = "taskA", frames: int = 20) -> None:
    episode_dir = root / episode_id
    episode_dir.mkdir(parents=True)
    (episode_dir / "meta.json").write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "task_id": task_id,
                "language": "toy task",
                "fps": 10,
                "num_frames": frames,
                "cameras": ["cam0"],
                "action_dim": 4,
                "proprio_dim": 4,
            }
        ),
        encoding="utf-8",
    )
    (episode_dir / "labels.json").write_text(
        json.dumps(
            {
                "primitive_boundaries": [
                    {"stage": "approach", "start": 0, "end": 9},
                    {"stage": "grasp", "start": 10, "end": 19},
                ],
                "success": True,
            }
        ),
        encoding="utf-8",
    )
    rng = np.random.default_rng(abs(hash(episode_id)) % 2**32)
    np.savez_compressed(
        episode_dir / "arrays.npz",
        proprio=rng.normal(size=(frames, 4)).astype(np.float32),
        action=rng.normal(size=(frames, 4)).astype(np.float32),
    )


def write_toy_features(root, episode_ids, frames: int = 20, dim: int = 8) -> None:
    for episode_id in episode_ids:
        write_feature_store(
            root / f"{episode_id}.npz",
            {"cam0": np.ones((frames, dim), dtype=np.float16)},
        )


def test_prepare_windows_writes_expected_files(tmp_path):
    episodes_root = tmp_path / "episodes"
    for i in range(5):
        write_toy_episode(episodes_root, f"ep{i}")
    output_dir = tmp_path / "windows"

    records = prepare_windows(
        episodes_root,
        output_dir,
        history=4,
        horizon=4,
        stride=2,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )

    assert records
    assert (output_dir / "windows.jsonl").exists()
    assert (output_dir / "labels.npz").exists()
    assert (output_dir / "index.json").exists()
    with np.load(output_dir / "labels.npz") as labels:
        assert labels["delta_phi"].shape[0] == len(records)
        assert labels["task_id"].dtype == np.int64


def test_prepare_windows_module_cli(tmp_path):
    episodes_root = tmp_path / "episodes"
    for i in range(5):
        write_toy_episode(episodes_root, f"ep{i}")
    output_dir = tmp_path / "windows"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp0.prepare_windows",
            "--episodes",
            str(episodes_root),
            "--output",
            str(output_dir),
            "--history",
            "4",
            "--horizon",
            "4",
            "--stride",
            "2",
            "--train-ratio",
            "0.6",
            "--val-ratio",
            "0.2",
            "--test-ratio",
            "0.2",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "wrote" in result.stdout
    assert (output_dir / "windows.jsonl").exists()


def test_read_episode_specs_validates_arrays(tmp_path):
    episodes_root = tmp_path / "episodes"
    write_toy_episode(episodes_root, "ep0")

    specs = read_episode_specs(episodes_root)

    assert len(specs) == 1
    assert specs[0].meta.episode_id == "ep0"
    assert specs[0].meta.success is True


def test_prepared_window_dataset_reads_arrays_and_features(tmp_path):
    episodes_root = tmp_path / "episodes"
    episode_ids = [f"ep{i}" for i in range(5)]
    for episode_id in episode_ids:
        write_toy_episode(episodes_root, episode_id)
    features_root = tmp_path / "features"
    write_toy_features(features_root, episode_ids)
    windows_dir = tmp_path / "windows"
    prepare_windows(
        episodes_root,
        windows_dir,
        history=4,
        horizon=4,
        stride=2,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )

    dataset = PreparedWindowDataset(
        windows_dir=windows_dir,
        episodes_dir=episodes_root,
        features_dir=features_root,
        split="train",
        feature_dim=8,
    )
    sample = dataset[0]
    batch = collate_batch([dataset[0], dataset[1]])

    assert sample["obs_features"].shape == (4, 1, 8)
    assert sample["proprio"].shape == (4,)
    assert sample["action_chunk"].shape == (4, 4)
    assert batch["obs_features"].shape == (2, 4, 1, 8)


def test_train_can_use_prepared_window_dataset(tmp_path):
    episodes_root = tmp_path / "episodes"
    episode_ids = [f"ep{i}" for i in range(5)]
    for episode_id in episode_ids:
        write_toy_episode(episodes_root, episode_id)
    features_root = tmp_path / "features"
    write_toy_features(features_root, episode_ids)
    windows_dir = tmp_path / "windows"
    prepare_windows(
        episodes_root,
        windows_dir,
        history=4,
        horizon=4,
        stride=2,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )
    output_dir = tmp_path / "outputs"
    config = apply_overrides(
        load_config("mvp0/configs/debug.yaml"),
        [
            "experiment=obs_action_stage_cf",
            "train.max_epochs=1",
            "data.batch_size=4",
            "data.horizon=4",
            "data.action_dim=4",
            "data.proprio_dim=4",
            f"data.windows_dir={windows_dir}",
            f"data.episodes_dir={episodes_root}",
            f"data.features_dir={features_root}",
            "features.feature_dim=8",
            "model.hidden_dim=32",
            "model.transformer_layers=1",
            "model.transformer_heads=4",
            f"output_dir={output_dir}",
        ],
    )

    metrics = train(config)

    assert "ranking_acc" in metrics
    assert (output_dir / "obs_action_stage_cf" / "best.pt").exists()


def test_mock_feature_extraction_cli(tmp_path):
    episodes_root = tmp_path / "episodes"
    for i in range(2):
        write_toy_episode(episodes_root, f"ep{i}")
    features_root = tmp_path / "features"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp0.extract_vision_features",
            "--episodes",
            str(episodes_root),
            "--output",
            str(features_root),
            "--feature-dim",
            "8",
            "--mock",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "wrote mock features" in result.stdout
    with np.load(features_root / "ep0.npz") as features:
        assert features["cam0"].shape == (20, 8)


def test_image_paths_for_camera_are_sorted(tmp_path):
    camera_dir = tmp_path / "ep0" / "images" / "cam0"
    camera_dir.mkdir(parents=True)
    (camera_dir / "000002.jpg").write_bytes(b"x")
    (camera_dir / "000001.png").write_bytes(b"x")
    (camera_dir / "ignore.txt").write_text("x", encoding="utf-8")

    paths = image_paths_for_camera(tmp_path / "ep0", "cam0")

    assert [path.name for path in paths] == ["000001.png", "000002.jpg"]
