from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ppwam.import_rh20t import import_rh20t_subset, load_scene_signals
from ppwam.import_rh20t import prompt_text_for_task
from ppwam.prompts import load_prompt_feature_store, read_prompt_table


CAMERA = "036422060215"


def _write_fake_scene(root: Path, scene_dir: str, n: int = 80) -> None:
    scene = root / scene_dir
    (scene / f"cam_{CAMERA}").mkdir(parents=True)
    (scene / "transformed").mkdir(parents=True)
    timestamps = np.arange(n, dtype=np.int64) * 40 + 1_000
    np.save(scene / f"cam_{CAMERA}" / "timestamps.npy", {"color": timestamps.tolist(), "depth": timestamps.tolist()})
    (scene / "metadata.json").write_text(json.dumps({"rating": 9, "calib_quality": 1}), encoding="utf-8")

    tcp_rows = []
    ft_rows = []
    gripper = {}
    for i, ts in enumerate(timestamps.tolist()):
        x = i * 0.01
        tcp_rows.append(
            {
                "timestamp": int(ts),
                "tcp": np.asarray([x, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            }
        )
        force = np.asarray([0.1, 0.1, 0.1], dtype=np.float64)
        if 20 <= i <= 55:
            force = np.asarray([4.0, 0.0, 0.0], dtype=np.float64)
        ft_rows.append({"timestamp": int(ts), "zeroed": np.concatenate([force, np.zeros(3)])})
        gripper[int(ts)] = {"gripper_info": [float(90 - i * 0.1), 0, 1]}
    payload = {CAMERA: tcp_rows}
    np.save(scene / "transformed" / "tcp_base.npy", payload)
    np.save(scene / "transformed" / "force_torque_base.npy", {CAMERA: ft_rows})
    np.save(scene / "transformed" / "gripper.npy", {CAMERA: gripper})


def test_load_scene_signals_builds_tcp_force_proxy(tmp_path: Path) -> None:
    _write_fake_scene(tmp_path, "task_0001_user_0001_scene_0001_cfg_0002")

    signals = load_scene_signals(tmp_path / "task_0001_user_0001_scene_0001_cfg_0002", camera=CAMERA)

    assert signals.proprio.shape == (80, 12)
    assert signals.action.shape == (80, 12)
    assert signals.contact_mask.any()
    assert signals.intervals


def test_import_rh20t_subset_writes_wam_episode_and_prompts(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "episodes"
    prompts = tmp_path / "prompts"
    _write_fake_scene(source, "task_0001_user_0001_scene_0001_cfg_0002")
    scene_list = tmp_path / "scenes.json"
    scene_list.write_text(
        json.dumps(
            {
                "selected_scenes": [
                    {
                        "scene_dir": "task_0001_user_0001_scene_0001_cfg_0002",
                        "task_id": "task_0001",
                        "task_description": "Turn the knob.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    imported = import_rh20t_subset(
        source_root=source,
        output=output,
        scene_list_json=scene_list,
        prompt_output=prompts,
        prompt_feature_dim=16,
        overwrite=True,
    )

    assert len(imported) == 1
    episode = output / imported[0].episode_id
    meta = json.loads((episode / "meta.json").read_text(encoding="utf-8"))
    labels = json.loads((episode / "labels.json").read_text(encoding="utf-8"))
    assert meta["source"] == "rh20t"
    assert meta["action_dim"] == 12
    assert labels["primitive_boundaries"]
    with np.load(episode / "arrays.npz") as arrays:
        assert arrays["proprio"].shape == (80, 12)
        assert arrays["action"].shape == (80, 12)
        assert "force_mag" in arrays

    table = read_prompt_table(prompts / "prompt_table.jsonl")
    features = load_prompt_feature_store(prompts / "prompt_features.npz", expected_dim=16)
    assert table[0].task_id == "task_0001"
    assert table[0].task_meta_text == "Turn the knob."
    assert table[0].primitive_chain == (
        "approach interaction target",
        "establish grasp or contact",
        "move or manipulate target",
        "release or finish interaction",
    )
    assert "task_0001" in features


def test_rh20t_prompt_text_uses_readable_fallback() -> None:
    assert prompt_text_for_task("task_0001", "") == "RH20T manipulation task task_0001."
    assert prompt_text_for_task("task_0001", "task_0001") == "RH20T manipulation task task_0001."
    assert prompt_text_for_task("task_0001", "Turn the knob.") == "Turn the knob."
