import json
import subprocess
import sys

import torch

from mvp0.config import apply_overrides, load_config
from mvp0.train import train


def _toy_config(tmp_path, experiment: str):
    return apply_overrides(
        load_config("mvp0/configs/debug.yaml"),
        [
            f"experiment={experiment}",
            "train.max_epochs=1",
            "data.num_samples=16",
            "data.batch_size=8",
            "model.hidden_dim=32",
            "model.transformer_layers=1",
            "model.transformer_heads=4",
            f"output_dir={tmp_path}",
        ],
    )


def test_time_prior_toy_training_writes_checkpoint(tmp_path):
    config = _toy_config(tmp_path, "time_prior")

    metrics = train(config)

    assert "delta_phi_mae" in metrics
    assert (tmp_path / "time_prior" / "best.pt").exists()
    assert (tmp_path / "time_prior" / "metrics.json").exists()


def test_obs_action_stage_cf_toy_training_writes_ranking_metrics(tmp_path):
    config = _toy_config(tmp_path, "obs_action_stage_cf")

    metrics = train(config)
    checkpoint = torch.load(tmp_path / "obs_action_stage_cf" / "best.pt", map_location="cpu", weights_only=False)
    saved_metrics = json.loads((tmp_path / "obs_action_stage_cf" / "metrics.json").read_text())

    assert "ranking_acc" in metrics
    assert "model_state" in checkpoint
    assert saved_metrics["ranking_acc"] == metrics["ranking_acc"]


def test_eval_and_plot_write_action_sensitivity_outputs(tmp_path):
    config = _toy_config(tmp_path, "obs_action_stage_cf")
    train(config)
    checkpoint = tmp_path / "obs_action_stage_cf" / "best.pt"
    eval_dir = tmp_path / "obs_action_stage_cf" / "eval"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp0.eval",
            "--checkpoint",
            str(checkpoint),
            "--split",
            "test",
            "--output",
            str(eval_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp0.plot",
            "--eval",
            str(eval_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    metrics = json.loads((eval_dir / "metrics.json").read_text())
    assert (eval_dir / "predictions.jsonl").exists()
    assert (eval_dir / "action_sensitivity.csv").exists()
    assert (eval_dir / "plots" / "delta_phi_hist.png").exists()
    assert (eval_dir / "plots" / "action_margin_hist.png").exists()
    assert "zero_ranking_acc" in metrics
