import json
from pathlib import Path

import numpy as np

from ppwam.potential_gain_audit import audit_potential_gain
from ppwam.prepare_windows import prepare_windows


def _write_toy_episode(root: Path, episode_id: str, frames: int = 20) -> None:
    episode_dir = root / episode_id
    episode_dir.mkdir(parents=True)
    (episode_dir / "meta.json").write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "task_id": "taskA",
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


def test_potential_gain_audit_writes_metrics_and_groups(tmp_path: Path) -> None:
    episodes_root = tmp_path / "episodes"
    for index in range(4):
        _write_toy_episode(episodes_root, f"ep{index}")
    windows_dir = tmp_path / "windows"
    prepare_windows(
        episodes_root,
        windows_dir,
        history=4,
        horizon=4,
        stride=2,
        train_ratio=0.5,
        val_ratio=0.25,
        test_ratio=0.25,
    )
    output_dir = tmp_path / "audit"

    metrics = audit_potential_gain(windows_dir, output_dir=output_dir)

    assert metrics["num_windows"] > 0
    assert metrics["has_phi_t"] == 1.0
    assert metrics["has_phi_future"] == 1.0
    assert metrics["has_delta_phi_raw"] == 1.0
    assert metrics["phi_consistency_abs_error_mean"] < 1.0e-6
    assert metrics["delta_phi_raw_positive_rate"] > 0.0
    assert (output_dir / "metrics.json").exists()
    assert (output_dir / "by_split_source_stage.csv").exists()
    loaded = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert loaded["num_windows"] == metrics["num_windows"]

    with np.load(windows_dir / "labels.npz") as labels:
        assert np.allclose(labels["delta_phi_raw"], labels["phi_future"] - labels["phi_t"])
