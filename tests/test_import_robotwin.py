from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from ppwam.import_robotwin import import_robotwin_hdf5
from ppwam.prepare_windows import prepare_windows


JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01fake\xff\xd9"


def _write_fake_robotwin_hdf5(path: Path, frames: int = 12) -> None:
    path.parent.mkdir(parents=True)
    with h5py.File(path, "w") as handle:
        joint = handle.create_group("joint_action")
        joint.create_dataset("vector", data=np.arange(frames * 14, dtype=np.float64).reshape(frames, 14))
        endpose = handle.create_group("endpose")
        endpose.create_dataset("left_endpose", data=np.ones((frames, 7), dtype=np.float64))
        endpose.create_dataset("right_endpose", data=np.ones((frames, 7), dtype=np.float64) * 2.0)
        endpose.create_dataset("left_gripper", data=np.ones(frames, dtype=np.float64))
        endpose.create_dataset("right_gripper", data=np.zeros(frames, dtype=np.float64))
        obs = handle.create_group("observation")
        for camera in ("head_camera", "right_camera"):
            group = obs.create_group(camera)
            group.create_dataset("rgb", data=[JPEG.ljust(32, b"\0") for _ in range(frames)], dtype="S32")


def test_import_robotwin_hdf5_writes_wam_episode_and_windows(tmp_path: Path) -> None:
    raw = tmp_path / "RoboTwin" / "data" / "beat_block_hammer" / "ppwam_smoke_clean" / "data" / "episode0.hdf5"
    _write_fake_robotwin_hdf5(raw, frames=16)
    instructions = raw.parent.parent / "instructions" / "episode0.json"
    instructions.parent.mkdir(parents=True)
    instructions.write_text(json.dumps({"seen": ["Hit the block with the hammer."]}), encoding="utf-8")
    output = tmp_path / "episodes"

    imported = import_robotwin_hdf5(
        [raw],
        output=output,
        instructions=instructions,
        overwrite=True,
    )

    assert len(imported) == 1
    episode = output / imported[0].episode_id
    meta = json.loads((episode / "meta.json").read_text(encoding="utf-8"))
    labels = json.loads((episode / "labels.json").read_text(encoding="utf-8"))
    assert meta["source"] == "robotwin"
    assert meta["task_id"] == "beat_block_hammer"
    assert meta["language"] == "Hit the block with the hammer."
    assert meta["proprio_dim"] == 30
    assert meta["action_dim"] == 30
    assert labels["label_source"] == "robotwin_linear_progress_v0"
    assert len(labels["potential"]) == 16
    assert labels["primitive_boundaries"][0] == {"stage": "approach", "start": 0, "end": 3}
    assert (episode / "images" / "head_camera" / "000000.jpg").read_bytes() == JPEG

    with np.load(episode / "arrays.npz") as arrays:
        assert arrays["proprio"].shape == (16, 30)
        assert arrays["action"].shape == (16, 30)
        np.testing.assert_allclose(arrays["action"][0], arrays["proprio"][1])
        assert arrays["joint_action_vector"].shape == (16, 14)

    records = prepare_windows(
        output,
        tmp_path / "windows",
        history=4,
        horizon=4,
        stride=2,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
    )
    assert records
    assert all(record.source == "robotwin" for record in records)
    assert all(record.phi_future >= record.phi_t for record in records)


def test_import_robotwin_hdf5_can_skip_images_and_cap_frames(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "task_a" / "config_a" / "data" / "episode2.hdf5"
    _write_fake_robotwin_hdf5(raw, frames=10)

    imported = import_robotwin_hdf5(
        [raw],
        output=tmp_path / "episodes",
        task_id="task_a",
        max_frames=6,
        skip_images=True,
        overwrite=True,
    )

    episode = Path(imported[0].output_dir)
    meta = json.loads((episode / "meta.json").read_text(encoding="utf-8"))
    assert meta["num_frames"] == 6
    assert meta["images_imported"] is False
    assert not (episode / "images").exists()


def test_import_robotwin_hdf5_uses_label_sidecar_for_negative_gain(tmp_path: Path) -> None:
    raw = tmp_path / "RoboTwin" / "data" / "beat_block_hammer" / "subsuccess" / "data" / "episode0.hdf5"
    _write_fake_robotwin_hdf5(raw, frames=16)
    sidecar = tmp_path / "sidecar.json"
    sidecar.write_text(
        json.dumps(
            {
                "label_source": "robotwin_subsuccess_rule_v0",
                "success": True,
                "suboptimal_type": "overshoot",
                "primitive_boundaries": [
                    {"stage": "approach", "start": 0, "end": 5},
                    {"stage": "move", "start": 6, "end": 15},
                ],
                "potential": [
                    0.00,
                    0.10,
                    0.20,
                    0.55,
                    0.56,
                    0.57,
                    0.58,
                    0.12,
                    0.13,
                    0.14,
                    0.60,
                    0.70,
                    0.80,
                    0.90,
                    0.95,
                    1.00,
                ],
            }
        ),
        encoding="utf-8",
    )

    imported = import_robotwin_hdf5(
        [raw],
        output=tmp_path / "episodes",
        label_sidecar=sidecar,
        skip_images=True,
        overwrite=True,
    )

    episode = Path(imported[0].output_dir)
    labels = json.loads((episode / "labels.json").read_text(encoding="utf-8"))
    assert labels["label_source"] == "robotwin_subsuccess_rule_v0"
    assert labels["suboptimal_type"] == "overshoot"
    assert labels["params"]["potential"] == "sidecar"

    records = prepare_windows(
        tmp_path / "episodes",
        tmp_path / "windows",
        history=4,
        horizon=4,
        stride=1,
        train_ratio=1.0,
        val_ratio=0.0,
        test_ratio=0.0,
    )

    assert any(record.delta_phi_raw < 0.0 for record in records)
    with np.load(tmp_path / "windows" / "labels.npz") as arrays:
        assert np.any(arrays["delta_phi_raw"] < 0.0)
