import json

import pytest
import torch

from ppwam.data import MockDatasetConfig, collate_batch
from ppwam.joint_flow import (
    EXPERIMENT,
    JointFlowDiT,
    PhiOnlyFlowCritic,
    MockJointFlowDataset,
    grouped_negative_metrics,
    make_flow_batch,
    score_action,
    run_full_joint_flow,
    source_stratified_metrics,
)
from ppwam.joint_flow_mvp16_report import top1_metrics


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


def test_joint_flow_dit_forward_shapes_with_phi_trajectory():
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
                    seed=8,
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
        layers=1,
        heads=4,
        history=3,
        horizon=5,
        phi_tokens=5,
    )
    flow = make_flow_batch(batch, phi_tokens=5)

    outputs = model(
        batch["obs_features"],
        batch["proprio_history"],
        batch["prompt_features"],
        flow.future_obs_noisy,
        flow.action_noisy,
        flow.phi_noisy,
        flow.tau,
    )

    assert flow.phi_target.shape == (2, 5)
    assert outputs["v_phi"].shape == (2, 5)


def test_phi_only_flow_critic_forward_shapes():
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
                    seed=18,
                )
            )[i]
            for i in range(2)
        ]
    )
    model = PhiOnlyFlowCritic(
        feature_dim=16,
        prompt_dim=10,
        proprio_dim=6,
        action_dim=8,
        hidden_dim=32,
        layers=1,
        heads=4,
        history=3,
        horizon=5,
        phi_tokens=5,
    )
    flow = make_flow_batch(batch, action_is_condition=True, phi_tokens=5)

    outputs = model(
        batch["obs_features"],
        batch["proprio_history"],
        batch["prompt_features"],
        flow.future_obs_noisy,
        flow.action_noisy,
        flow.phi_noisy,
        flow.tau,
        action_is_condition=True,
    )

    assert outputs["v_obs"].shape == (2, 5, 16)
    assert outputs["v_action"].shape == (2, 5, 8)
    assert outputs["v_phi"].shape == (2, 5)
    assert torch.count_nonzero(outputs["v_obs"]) == 0
    assert torch.count_nonzero(outputs["v_action"]) == 0


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


def test_make_flow_batch_phi_trajectory_ends_at_delta_phi():
    batch = collate_batch(
        [
            MockJointFlowDataset(
                MockDatasetConfig(num_samples=2, history=3, horizon=5, cameras=2, feature_dim=16, action_dim=8)
            )[i]
            for i in range(2)
        ]
    )

    flow = make_flow_batch(batch, action_is_condition=False, phi_tokens=5)

    assert flow.phi_target.shape == (2, 5)
    torch.testing.assert_close(flow.phi_target[:, -1], batch["delta_phi"])
    assert torch.all(flow.phi_target[:, 1:] >= flow.phi_target[:, :-1])


def test_score_action_supports_multistep_phi_trajectory():
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
                    seed=9,
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
        layers=1,
        heads=4,
        history=3,
        horizon=5,
        phi_tokens=5,
    )

    score = score_action(model, batch, denoise_steps=3, phi_tokens=5)

    assert score.shape == (2,)


def test_grouped_negative_metrics_are_tie_aware_and_grouped():
    pos = torch.tensor([0.7, 0.5, 0.2, 0.8, 0.3, 0.4])
    neg = torch.tensor([0.1, 0.5, 0.6, 0.7, 0.4, 0.4])
    negative_types = ["zero", "wrong_arm", "scaled_0.25", "scaled_1.75", "reverse", "shuffle"]

    metrics = grouped_negative_metrics(pos, neg, negative_types)

    assert metrics["coarse_action_cf_ranking_acc"] == pytest.approx((1.0 + 0.5 + 0.0 + 1.0) / 4.0)
    assert metrics["coarse_action_cf_mean_margin"] == pytest.approx(((0.6 + 0.0 - 0.4 + 0.1) / 4.0))
    assert metrics["temporal_diagnostic_ranking_acc"] == pytest.approx((0.0 + 0.5) / 2.0)
    assert metrics["temporal_diagnostic_mean_margin"] == pytest.approx((-0.1 + 0.0) / 2.0)


def test_source_stratified_metrics_report_per_source_ranking():
    pred = torch.tensor([0.9, 0.2, 0.6, 0.4])
    target = torch.tensor([1.0, 0.0, 0.5, 0.5])
    source_id = torch.tensor([0, 0, 1, 1])
    pos = torch.tensor([0.9, 0.6, 0.4, 0.8])
    neg = torch.tensor([0.1, 0.7, 0.4, 0.2])
    negative_source_id = torch.tensor([0, 0, 1, 1])
    negative_types = ["zero", "reverse", "zero", "scaled_0.25"]

    metrics = source_stratified_metrics(
        pred,
        target,
        source_id,
        source_names={0: "gm100", 1: "rh20t"},
        pos_delta_phi=pos,
        neg_delta_phi=neg,
        negative_types=negative_types,
        negative_source_ids=negative_source_id,
    )

    assert metrics["source_gm100_num_windows"] == 2.0
    assert metrics["source_gm100_delta_phi_mae"] == pytest.approx(0.15)
    assert metrics["source_gm100_all_negatives_tie_aware_ranking_acc"] == pytest.approx(0.5)
    assert metrics["source_gm100_zero_ranking_acc"] == pytest.approx(1.0)
    assert metrics["source_gm100_temporal_diagnostic_ranking_acc"] == pytest.approx(0.0)
    assert metrics["source_rh20t_delta_phi_rmse"] == pytest.approx(0.1)
    assert metrics["source_rh20t_all_negatives_tie_aware_ranking_acc"] == pytest.approx(0.75)
    assert metrics["source_rh20t_coarse_action_cf_ranking_acc"] == pytest.approx(0.75)


def test_top1_metrics_use_best_negative_per_candidate_set(tmp_path):
    path = tmp_path / "action_sensitivity.csv"
    path.write_text(
        "\n".join(
            [
                "index,negative_type,pos_delta_phi,neg_delta_phi,margin,is_correct",
                "0,zero,0.8,0.2,0.6,1",
                "0,wrong_arm,0.8,0.7,0.1,1",
                "0,reverse,0.8,0.9,-0.1,0",
                "1,zero,0.4,0.1,0.3,1",
                "1,wrong_arm,0.4,0.6,-0.2,0",
                "1,shuffle,0.4,0.3,0.1,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    metrics = top1_metrics(path)

    assert metrics["all_negatives_top1_acc"] == pytest.approx(0.0)
    assert metrics["all_negatives_top1_margin"] == pytest.approx((-0.1 - 0.2) / 2.0)
    assert metrics["coarse_action_cf_top1_acc"] == pytest.approx(0.5)
    assert metrics["coarse_action_cf_top1_margin"] == pytest.approx((0.1 - 0.2) / 2.0)
    assert metrics["temporal_diagnostic_top1_acc"] == pytest.approx(0.5)
    assert metrics["temporal_diagnostic_top1_margin"] == pytest.approx((-0.1 + 0.1) / 2.0)


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
            "phi_tokens": 5,
        },
        "loss": {
            "obs_weight": 1.0,
            "action_weight": 1.0,
            "phi_weight": 2.0,
            "counterfactual_weight": 0.01,
            "critic_flow_weight": 0.5,
            "margin": 0.01,
        },
        "score": {"denoise_steps": 2, "train_denoise_steps": 1, "phi_reduce": "last"},
        "negatives": {"train_types": ["zero", "reverse"]},
        "eval": {"negative_types": "zero,reverse"},
        "optim": {"lr": 1.0e-3, "weight_decay": 0.0, "grad_clip_norm": 1.0},
        "train": {
            "max_epochs": 1,
            "save_best_by": "val/delta_phi_mae",
            "action_condition_prob": 0.75,
            "cf_negatives_per_batch": "all",
        },
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
    assert "coarse_action_cf_ranking_acc" in loaded
    assert "temporal_diagnostic_ranking_acc" in loaded


def test_phi_only_flow_critic_smoke_run(tmp_path):
    config = {
        "seed": 321,
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
            "name": "phi_only_flow_critic_test",
            "hidden_dim": 32,
            "transformer_layers": 1,
            "transformer_heads": 4,
            "dropout": 0.0,
            "mlp_ratio": 2,
            "phi_tokens": 5,
        },
        "loss": {
            "obs_weight": 0.0,
            "action_weight": 0.0,
            "phi_weight": 2.0,
            "counterfactual_weight": 0.01,
            "critic_flow_weight": 0.0,
            "margin": 0.01,
        },
        "score": {"denoise_steps": 2, "train_denoise_steps": 1, "phi_reduce": "last"},
        "negatives": {"train_types": ["zero", "wrong_arm"]},
        "eval": {"negative_types": "zero,wrong_arm"},
        "optim": {"lr": 1.0e-3, "weight_decay": 0.0, "grad_clip_norm": 1.0},
        "train": {
            "max_epochs": 1,
            "save_best_by": "val/delta_phi_mae",
            "action_condition_prob": 1.0,
            "cf_negatives_per_batch": "all",
        },
    }

    metrics = run_full_joint_flow(config)
    run_dir = tmp_path / "outputs" / EXPERIMENT

    assert "delta_phi_mae" in metrics
    assert (run_dir / "best.pt").exists()
    loaded = json.loads((run_dir / "eval_test" / "metrics.json").read_text(encoding="utf-8"))
    assert "all_negatives_tie_aware_ranking_acc" in loaded
