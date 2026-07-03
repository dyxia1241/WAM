import json
import subprocess
import sys

import numpy as np

from ppwam.data import PreparedWindowDataset, collate_batch
from ppwam.features import write_feature_store
from ppwam.prepare_windows import prepare_windows, read_episode_metas, read_episode_specs
from ppwam.config import apply_overrides, load_config
from ppwam.extract_vision_features import image_paths_for_camera
from ppwam.norm_stats import compute_norm_stats
from ppwam.prompts import PromptRecord, write_prompt_feature_store
from ppwam.train import train


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


def write_toy_prompt_features(path, task_ids, dim: int = 12) -> dict[str, np.ndarray]:
    records = [
        PromptRecord(
            task_id=task_id,
            task_meta_text=f"meta for {task_id}",
            primitive_chain=("grasp object", "place object"),
            prompt=f"prompt for {task_id}",
        )
        for task_id in task_ids
    ]
    features = np.arange(len(records) * dim, dtype=np.float32).reshape(len(records), dim)
    write_prompt_feature_store(path, records, features)
    return {task_id: features[index] for index, task_id in enumerate(task_ids)}


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
            "ppwam.prepare_windows",
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


def test_prepared_window_dataset_reads_prompt_features_by_raw_task_id(tmp_path):
    episodes_root = tmp_path / "episodes"
    episode_ids = [f"ep{i}" for i in range(6)]
    for index, episode_id in enumerate(episode_ids):
        write_toy_episode(episodes_root, episode_id, task_id=f"task{index % 2}")
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
    prompt_feature_path = tmp_path / "prompt_features.npz"
    expected = write_toy_prompt_features(prompt_feature_path, ["task0", "task1"], dim=12)

    dataset = PreparedWindowDataset(
        windows_dir=windows_dir,
        episodes_dir=episodes_root,
        features_dir=features_root,
        split="train",
        feature_dim=8,
        prompt_features=prompt_feature_path,
        prompt_feature_dim=12,
    )
    sample = dataset[0]
    raw_task_id = dataset.windows[dataset.indices[0]]["task_id"]

    assert sample["prompt_features"].shape == (12,)
    np.testing.assert_allclose(sample["prompt_features"].numpy(), expected[raw_task_id])


def test_compute_norm_stats_uses_train_split_only(tmp_path):
    episodes_root = tmp_path / "episodes"
    episode_ids = [f"ep{i}" for i in range(5)]
    for index, episode_id in enumerate(episode_ids):
        write_toy_episode(episodes_root, episode_id, frames=20)
        action = np.full((20, 4), index + 1, dtype=np.float32)
        proprio = np.full((20, 4), (index + 1) * 10, dtype=np.float32)
        np.savez_compressed(episodes_root / episode_id / "arrays.npz", proprio=proprio, action=action)
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
        seed=123,
    )

    output = windows_dir / "norm_stats.json"
    stats = compute_norm_stats(windows_dir=windows_dir, episodes_dir=episodes_root, output=output)
    index = json.loads((windows_dir / "index.json").read_text())
    train_ids = index["split"]["train"]
    expected_action = np.concatenate(
        [np.load(episodes_root / episode_id / "arrays.npz")["action"] for episode_id in train_ids],
        axis=0,
    )
    expected_proprio = np.concatenate(
        [np.load(episodes_root / episode_id / "arrays.npz")["proprio"] for episode_id in train_ids],
        axis=0,
    )

    assert output.exists()
    assert stats["counts"]["episodes"] == len(train_ids)
    assert stats["counts"]["frames"] == expected_action.shape[0]
    np.testing.assert_allclose(stats["action"]["mean"], expected_action.mean(axis=0))
    np.testing.assert_allclose(stats["proprio"]["mean"], expected_proprio.mean(axis=0))


def test_prepared_window_dataset_applies_norm_stats(tmp_path):
    episodes_root = tmp_path / "episodes"
    episode_ids = [f"ep{i}" for i in range(5)]
    for index, episode_id in enumerate(episode_ids):
        write_toy_episode(episodes_root, episode_id, frames=20)
        ramp = np.arange(80, dtype=np.float32).reshape(20, 4)
        np.savez_compressed(
            episodes_root / episode_id / "arrays.npz",
            proprio=ramp + index * 100,
            action=ramp + index * 10,
        )
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
        seed=123,
    )
    norm_stats = compute_norm_stats(windows_dir=windows_dir, episodes_dir=episodes_root)

    dataset = PreparedWindowDataset(
        windows_dir=windows_dir,
        episodes_dir=episodes_root,
        features_dir=features_root,
        split="train",
        feature_dim=8,
        norm_stats=norm_stats,
    )
    label_index = dataset.indices[0]
    window = dataset.windows[label_index]
    with np.load(episodes_root / window["episode_id"] / "arrays.npz") as arrays:
        raw_proprio = arrays["proprio"][window["t"]]
        raw_action = arrays["action"][window["future_indices"]]
    sample = dataset[0]

    action_mean = np.asarray(norm_stats["action"]["mean"], dtype=np.float32)
    action_std = np.asarray(norm_stats["action"]["std"], dtype=np.float32)
    proprio_mean = np.asarray(norm_stats["proprio"]["mean"], dtype=np.float32)
    proprio_std = np.asarray(norm_stats["proprio"]["std"], dtype=np.float32)
    np.testing.assert_allclose(sample["proprio"].numpy(), (raw_proprio - proprio_mean) / proprio_std)
    np.testing.assert_allclose(sample["action_chunk"].numpy(), (raw_action - action_mean) / action_std)


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
        load_config("configs/debug.yaml"),
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
            "ppwam.extract_vision_features",
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


def test_mock_feature_extraction_supports_label_free_episodes(tmp_path):
    episodes_root = tmp_path / "episodes"
    episode_dir = episodes_root / "gm100_ep0"
    episode_dir.mkdir(parents=True)
    (episode_dir / "meta.json").write_text(
        json.dumps(
            {
                "episode_id": "gm100_ep0",
                "task_id": "task_00001",
                "language": "hit-ball",
                "fps": 30,
                "num_frames": 6,
                "cameras": ["camera_top", "camera_wrist_left"],
                "action_dim": 14,
                "proprio_dim": 14,
            }
        ),
        encoding="utf-8",
    )
    features_root = tmp_path / "features"

    metas = read_episode_metas(episodes_root)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ppwam.extract_vision_features",
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

    assert metas[0].episode_id == "gm100_ep0"
    assert "wrote mock features" in result.stdout
    with np.load(features_root / "gm100_ep0.npz") as features:
        assert features["camera_top"].shape == (6, 8)
        assert features["camera_wrist_left"].shape == (6, 8)


def test_image_paths_for_camera_are_sorted(tmp_path):
    camera_dir = tmp_path / "ep0" / "images" / "cam0"
    camera_dir.mkdir(parents=True)
    (camera_dir / "000002.jpg").write_bytes(b"x")
    (camera_dir / "000001.png").write_bytes(b"x")
    (camera_dir / "ignore.txt").write_text("x", encoding="utf-8")

    paths = image_paths_for_camera(tmp_path / "ep0", "cam0")

    assert [path.name for path in paths] == ["000001.png", "000002.jpg"]
