import json

import torch

from mvp0.data import MockDatasetConfig, collate_batch
from mvp0.joint_flow import (
    EXPERIMENT,
    JointFlowDiT,
    MockJointFlowDataset,
    make_flow_batch,
    run_full_joint_flow,
)


def test_mock_joint_flow_dataset_returns_future_obs_and_proprio_history():
    dataset = MockJointFlowDataset(
        MockDatasetConfig(
            num_samples=4,
            history=3,
            horizon=5,
            cameras=2,
            feature_dim=16,
            proprio_dim=6,
            action_dim=8,
            prompt_dim=10,
        )
    )

    sample = dataset[0]

    assert sample["obs_features"].shape == (3, 2, 16)
    assert sample["future_obs_features"].shape == (5, 2, 16)
    assert sample["proprio_history"].shape == (3, 6)
    assert sample["action_chunk"].shape == (5, 8)
    assert sample["prompt_features"].shape == (10,)


def test_joint_flow_dit_forward_shapes():
    batch = collate_batch(
        [
            MockJointFlowDataset(
                MockDatasetConfig(
                    num_samples=2,
                    history=3,
                    horizon=5,
                    cameras=2,
                    feature_dim=16,
                    proprio_dim=6,
                    action_dim=8,
                    prompt_dim=10,
                    seed=7,
                )
            )[i]
            for i in range(2)
        ]
    )
    model = JointFlowDiT(
        feature_dim=16,
        prompt_dim=10,
        proprio_dim=6,
        action_dim=8,
        hidden_dim=32,
        layers=2,
        heads=4,
        history=3,
        horizon=5,
    )
    flow = make_flow_batch(batch)

    outputs = model(
        batch["obs_features"],
        batch["proprio_history"],
        batch["prompt_features"],
        flow.future_obs_noisy,
        flow.action_noisy,
        flow.phi_noisy,
        flow.tau,
    )

    assert outputs["v_obs"].shape == (2, 5, 16)
    assert outputs["v_action"].shape == (2, 5, 8)
    assert outputs["v_phi"].shape == (2, 1)


def test_make_flow_batch_matches_target_shapes():
    batch = collate_batch(
        [
            MockJointFlowDataset(
                MockDatasetConfig(num_samples=2, history=3, horizon=5, cameras=2, feature_dim=16, action_dim=8)
            )[i]
            for i in range(2)
        ]
    )

    flow = make_flow_batch(batch, action_is_condition=False)

    assert flow.future_obs_target.shape == (2, 5, 16)
    assert flow.action_target.shape == (2, 5, 8)
    assert flow.phi_target.shape == (2, 1)
    assert flow.v_obs_target.shape == flow.future_obs_noisy.shape
    assert flow.v_action_target.shape == flow.action_noisy.shape
    assert flow.v_phi_target.shape == flow.phi_noisy.shape


def test_joint_flow_smoke_run_writes_metrics_figures_and_report(tmp_path):
    config = {
        "seed": 123,
        "device": "cpu",
        "output_dir": str(tmp_path / "outputs"),
        "data": {
            "windows_dir": None,
            "history": 3,
            "horizon": 5,
            "batch_size": 8,
            "num_samples": 24,
            "cameras": 2,
            "prompt_feature_dim": 10,
            "proprio_dim": 6,
            "action_dim": 8,
        },
        "features": {"feature_dim": 16},
        "model": {
            "hidden_dim": 32,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "dropout": 0.0,
            "mlp_ratio": 2,
        },
        "loss": {
            "obs_weight": 1.0,
            "action_weight": 1.0,
            "phi_weight": 2.0,
            "counterfactual_weight": 0.01,
            "margin": 0.01,
        },
        "negatives": {"train_types": ["zero", "reverse"]},
        "eval": {"negative_types": "zero,reverse"},
        "optim": {"lr": 1.0e-3, "weight_decay": 0.0, "grad_clip_norm": 1.0},
        "train": {"max_epochs": 1, "save_best_by": "val/delta_phi_mae", "action_condition_prob": 0.5},
    }

    metrics = run_full_joint_flow(config)
    run_dir = tmp_path / "outputs" / EXPERIMENT

    assert "delta_phi_mae" in metrics
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "eval_test" / "metrics.json").exists()
    assert (run_dir / "eval_test" / "action_sensitivity.csv").exists()
    assert (run_dir / "figures" / "delta_phi_scatter.png").exists()
    assert (run_dir / "figures" / "action_margin_hist.png").exists()
    assert (run_dir / "experiment_report.md").exists()

    loaded = json.loads((run_dir / "eval_test" / "metrics.json").read_text(encoding="utf-8"))
    assert "all_negatives_tie_aware_ranking_acc" in loaded
