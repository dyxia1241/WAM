from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ppwam.data import MockDatasetConfig, collate_batch
from ppwam.joint_flow import JointFlowDiT, MockJointFlowDataset
from ppwam.joint_flow_sample_select import evaluate_sample_selection, sample_joint_futures


def test_sample_joint_futures_shapes() -> None:
    dataset = MockJointFlowDataset(
        MockDatasetConfig(
            num_samples=3,
            history=3,
            horizon=4,
            cameras=2,
            feature_dim=8,
            proprio_dim=5,
            action_dim=6,
            prompt_dim=7,
        )
    )
    batch = collate_batch([dataset[0], dataset[1]])
    model = JointFlowDiT(
        feature_dim=8,
        prompt_dim=7,
        proprio_dim=5,
        action_dim=6,
        hidden_dim=32,
        layers=1,
        heads=4,
        history=3,
        horizon=4,
        phi_tokens=4,
    )

    futures = sample_joint_futures(model, batch, num_samples=3, denoise_steps=2, phi_tokens=4)

    assert futures["action"].shape == (2, 3, 4, 6)
    assert futures["future_obs"].shape == (2, 3, 4, 8)
    assert futures["phi_state"].shape == (2, 3, 4)
    assert futures["generated_phi"].shape == (2, 3)
    assert torch.isfinite(futures["generated_phi"]).all()


def test_evaluate_sample_selection_writes_metrics(tmp_path: Path) -> None:
    config = {
        "seed": 42,
        "data": {"history": 3, "horizon": 4, "proprio_dim": 5, "action_dim": 6, "prompt_feature_dim": 7},
        "features": {"feature_dim": 8},
        "model": {"hidden_dim": 32, "transformer_layers": 1, "transformer_heads": 4, "phi_tokens": 4},
        "score": {"denoise_steps": 2, "phi_reduce": "last", "future_obs_init": "zero"},
    }
    dataset = MockJointFlowDataset(
        MockDatasetConfig(
            num_samples=4,
            history=3,
            horizon=4,
            cameras=2,
            feature_dim=8,
            proprio_dim=5,
            action_dim=6,
            prompt_dim=7,
        )
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_batch)
    model = JointFlowDiT(
        feature_dim=8,
        prompt_dim=7,
        proprio_dim=5,
        action_dim=6,
        hidden_dim=32,
        layers=1,
        heads=4,
        history=3,
        horizon=4,
        phi_tokens=4,
    )

    metrics = evaluate_sample_selection(
        model,
        loader,
        config,
        torch.device("cpu"),
        output_dir=tmp_path,
        num_samples=3,
        denoise_steps=2,
        max_batches=1,
    )

    assert metrics["num_examples"] == 2.0
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "selection_rows.csv").exists()
