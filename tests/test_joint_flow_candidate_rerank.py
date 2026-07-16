import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from ppwam.data import MockDatasetConfig, collate_batch
from ppwam.joint_flow import JointFlowDiT, MockJointFlowDataset
from ppwam.joint_flow_candidate_rerank import evaluate_candidate_reranking


def test_evaluate_candidate_reranking_writes_mixed_candidate_outputs(tmp_path: Path) -> None:
    dataset = MockJointFlowDataset(
        MockDatasetConfig(
            num_samples=16,
            history=3,
            horizon=5,
            cameras=2,
            feature_dim=12,
            proprio_dim=6,
            action_dim=8,
            prompt_dim=10,
            num_tasks=3,
            num_stages=4,
            seed=45,
        )
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)
    model = JointFlowDiT(
        feature_dim=12,
        prompt_dim=10,
        proprio_dim=6,
        action_dim=8,
        hidden_dim=32,
        layers=1,
        heads=4,
        history=3,
        horizon=5,
        phi_tokens=3,
    )
    config = {
        "seed": 45,
        "data": {"history": 3, "horizon": 5},
        "features": {"feature_dim": 12},
        "model": {"phi_tokens": 3},
        "score": {"denoise_steps": 1, "phi_reduce": "last", "future_obs_init": "zero"},
    }

    metrics = evaluate_candidate_reranking(
        scorer_model=model,
        scorer_config=config,
        loader=loader,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        generator_model=model,
        generator_config=config,
        hard_candidate_types=["same_task_phase_wrong", "cross_task"],
        perturb_sigmas=[0.1],
        num_wam_samples=2,
        max_anchors=6,
        rescore_batch_size=16,
    )

    assert metrics["num_anchors"] == 6.0
    assert metrics["candidate_count"] == pytest.approx(7.0)
    assert "model_phi_strict_pairwise_acc" in metrics
    assert "calibrated_smooth_strict_pairwise_acc" in metrics
    assert "calibrated_smooth_selected_logged_rate" in metrics
    assert (tmp_path / "candidate_rerank_metrics.json").exists()
    assert (tmp_path / "candidate_rerank_candidates.csv").exists()
    assert (tmp_path / "candidate_rerank_selection.csv").exists()
    candidate_types = json.loads((tmp_path / "candidate_types.json").read_text(encoding="utf-8"))
    assert candidate_types[:3] == ["logged", "same_task_phase_wrong", "cross_task"]
    assert "wam_sample_1" in candidate_types
