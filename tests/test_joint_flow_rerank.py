import json

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from ppwam.data import MockDatasetConfig, collate_batch
from ppwam.joint_flow import JointFlowDiT, MockJointFlowDataset
from ppwam.joint_flow_rerank import (
    CandidateBank,
    build_candidate_bank,
    evaluate_hard_reranking,
    reranking_metrics,
    select_hard_candidate_indices,
    top1_metrics,
)


def _toy_bank() -> CandidateBank:
    return CandidateBank(
        action_chunk=torch.zeros((5, 2, 4), dtype=torch.float32),
        task_id=np.asarray([0, 0, 0, 1, 1], dtype=np.int64),
        stage_id=np.asarray([0, 1, 0, 0, 2], dtype=np.int64),
        primitive_time=np.asarray([0.10, 0.15, 0.90, 0.12, 0.80], dtype=np.float32),
        delta_phi=np.asarray([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
        obs_embedding=np.asarray(
            [
                [0.0, 0.0],
                [0.1, 0.0],
                [3.0, 0.0],
                [0.2, 0.0],
                [4.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_select_hard_candidate_indices_uses_expected_constraints():
    selected = select_hard_candidate_indices(
        _toy_bank(),
        np.asarray([0], dtype=np.int64),
        candidate_types=[
            "same_task_phase_wrong",
            "same_task_far_progress",
            "cross_task",
            "nearest_obs_wrong_action",
        ],
        far_progress_threshold=0.35,
    )

    assert selected["same_task_phase_wrong"].tolist() == [1]
    assert selected["same_task_far_progress"].tolist() == [2]
    assert selected["cross_task"].tolist() == [3]
    assert selected["nearest_obs_wrong_action"].tolist() == [1]


def test_reranking_metrics_are_pairwise_and_top1():
    rows = [
        {
            "index": 0,
            "candidate_type": "cross_task",
            "source_index": 3,
            "positive_score": 0.8,
            "candidate_score": 0.2,
            "margin": 0.6,
            "pairwise_correct": 1,
        },
        {
            "index": 0,
            "candidate_type": "same_task_far_progress",
            "source_index": 2,
            "positive_score": 0.8,
            "candidate_score": 0.9,
            "margin": -0.1,
            "pairwise_correct": 0,
        },
        {
            "index": 1,
            "candidate_type": "cross_task",
            "source_index": 4,
            "positive_score": 0.5,
            "candidate_score": 0.5,
            "margin": 0.0,
            "pairwise_correct": 0,
        },
    ]
    selection_rows = [
        {
            "index": 0,
            "positive_score": 0.8,
            "best_negative_score": 0.9,
            "best_negative_type": "same_task_far_progress",
            "best_negative_source_index": 2,
            "positive_rank": 2,
            "strict_top1": 0,
            "tie_aware_top1_credit": 0.0,
            "candidate_count": 3,
        },
        {
            "index": 1,
            "positive_score": 0.5,
            "best_negative_score": 0.5,
            "best_negative_type": "cross_task",
            "best_negative_source_index": 4,
            "positive_rank": 1,
            "strict_top1": 0,
            "tie_aware_top1_credit": 0.5,
            "candidate_count": 2,
        },
    ]

    pairwise = reranking_metrics(rows)
    top1 = top1_metrics(selection_rows)

    assert pairwise["hard_pairwise_ranking_acc"] == pytest.approx(0.5)
    assert pairwise["cross_task_ranking_acc"] == pytest.approx(0.75)
    assert top1["hard_top1_acc"] == pytest.approx(0.0)
    assert top1["hard_tie_aware_top1_acc"] == pytest.approx(0.25)
    assert top1["hard_top3_acc"] == pytest.approx(1.0)


def test_evaluate_hard_reranking_writes_outputs(tmp_path):
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
            seed=44,
        )
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False, collate_fn=collate_batch)
    bank = build_candidate_bank(loader)
    assert len(bank) == 16

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
        "seed": 44,
        "data": {"history": 3, "horizon": 5},
        "features": {"feature_dim": 12},
        "model": {"phi_tokens": 3},
        "score": {"denoise_steps": 1, "phi_reduce": "last", "future_obs_init": "zero"},
    }
    out = tmp_path / "rerank"

    metrics = evaluate_hard_reranking(
        model,
        loader,
        config,
        torch.device("cpu"),
        output_dir=out,
        candidate_types=["same_task_phase_wrong", "cross_task"],
        max_anchors=6,
    )

    assert metrics["hard_num_anchors"] == 6.0
    assert "hard_pairwise_ranking_acc" in metrics
    assert (out / "hard_reranking_pairs.csv").exists()
    assert (out / "hard_reranking_selection.csv").exists()
    loaded = json.loads((out / "hard_reranking_metrics.json").read_text(encoding="utf-8"))
    assert loaded["hard_candidate_count_mean"] == pytest.approx(3.0)
