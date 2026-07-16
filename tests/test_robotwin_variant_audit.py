import json

import numpy as np

from ppwam.robotwin_variant_audit import audit_robotwin_variants, infer_task, infer_variant


def test_infer_task_and_variant_from_robotwin_episode_id():
    episode_id = "beat_block_hammer_overshoot_2x_v1_beat_block_hammer_episode0"

    assert infer_task(episode_id) == "beat_block_hammer"
    assert infer_variant(episode_id) == "overshoot"


def test_audit_robotwin_variants_groups_signed_gain(tmp_path):
    windows_dir = tmp_path / "windows"
    windows_dir.mkdir()
    windows = [
        {
            "episode_id": "beat_block_hammer_detour_2x_v1_beat_block_hammer_episode0",
            "task_id": "beat_block_hammer",
            "split": "train",
        },
        {
            "episode_id": "beat_block_hammer_detour_2x_v1_beat_block_hammer_episode0",
            "task_id": "beat_block_hammer",
            "split": "train",
        },
        {
            "episode_id": "click_bell_hesitation_2x_v1_click_bell_episode0",
            "task_id": "click_bell",
            "split": "val",
        },
    ]
    (windows_dir / "windows.jsonl").write_text(
        "\n".join(json.dumps(window) for window in windows) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(windows_dir / "labels.npz", delta_phi_raw=np.asarray([-0.2, 0.1, 0.0], dtype=np.float32))

    rows, overall = audit_robotwin_variants(windows_dir)

    assert overall["negative_rate"] == 1.0 / 3.0
    detour = next(row for row in rows if row["variant"] == "detour")
    assert detour["task"] == "beat_block_hammer"
    assert detour["negative_rate"] == 0.5
    assert detour["delta_phi_raw_min"] == np.float32(-0.2)
