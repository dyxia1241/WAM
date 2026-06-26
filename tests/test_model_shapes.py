import torch

from mvp0.data import MockDatasetConfig, MockWindowDataset, collate_batch
from mvp0.model import MLPCritic, StageFiLMTransformerCritic, TimePrior


def _batch():
    dataset = MockWindowDataset(
        MockDatasetConfig(
            num_samples=4,
            history=4,
            horizon=8,
            cameras=2,
            feature_dim=32,
            proprio_dim=6,
            action_dim=4,
            num_tasks=3,
        )
    )
    return collate_batch([dataset[i] for i in range(4)])


def test_time_prior_output_shape():
    batch = _batch()
    model = TimePrior(num_stages=5, num_tasks=3)

    output = model(batch["primitive_time"], batch["stage_id"], batch["task_id"])

    assert output.shape == (4, 1)


def test_mlp_critic_output_shape():
    batch = _batch()
    model = MLPCritic(
        feature_dim=32,
        proprio_dim=6,
        action_dim=4,
        horizon=8,
        num_tasks=3,
    )

    output = model(
        batch["obs_features"],
        batch["proprio"],
        batch["action_chunk"],
        batch["stage_id"],
        batch["task_id"],
    )

    assert output.shape == (4, 1)


def test_stage_film_transformer_output_shape():
    batch = _batch()
    model = StageFiLMTransformerCritic(
        feature_dim=32,
        proprio_dim=6,
        action_dim=4,
        num_tasks=3,
        hidden_dim=32,
        transformer_layers=1,
        transformer_heads=4,
        dropout=0.0,
    )

    output = model(
        batch["obs_features"],
        batch["proprio"],
        batch["action_chunk"],
        batch["stage_id"],
        batch["task_id"],
    )

    assert output.shape == (4, 1)
    assert torch.isfinite(output).all()

