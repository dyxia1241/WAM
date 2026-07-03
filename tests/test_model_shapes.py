import torch

from ppwam.data import MockDatasetConfig, MockWindowDataset, collate_batch
from ppwam.model import MLPCritic, PromptFiLMTransformerCritic, StageFiLMTransformerCritic, TimePrior
from ppwam.train import forward_model


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
            prompt_dim=16,
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


def test_prompt_film_transformer_output_shape():
    batch = _batch()
    model = PromptFiLMTransformerCritic(
        feature_dim=32,
        proprio_dim=6,
        action_dim=4,
        prompt_dim=16,
        hidden_dim=32,
        transformer_layers=1,
        transformer_heads=4,
        dropout=0.0,
        use_action=True,
    )

    output = model(
        batch["obs_features"],
        batch["proprio"],
        batch["action_chunk"],
        batch["prompt_features"],
    )

    assert output.shape == (4, 1)
    assert torch.isfinite(output).all()


def test_prompt_experiment_ignores_stage_and_numeric_task_ids():
    batch = _batch()
    model = PromptFiLMTransformerCritic(
        feature_dim=32,
        proprio_dim=6,
        action_dim=4,
        prompt_dim=16,
        hidden_dim=32,
        transformer_layers=1,
        transformer_heads=4,
        dropout=0.0,
        use_action=True,
    )
    model.eval()
    changed = {key: value.clone() for key, value in batch.items()}
    changed["stage_id"] = (changed["stage_id"] + 1) % 5
    changed["task_id"] = (changed["task_id"] + 1) % 3

    output = forward_model(model, batch, "obs_action_prompt")
    changed_output = forward_model(model, changed, "obs_action_prompt")

    torch.testing.assert_close(output, changed_output)


def test_joint_action_stage_ignores_visual_features():
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
    model.eval()
    changed = {key: value.clone() for key, value in batch.items()}
    changed["obs_features"] = changed["obs_features"] + 100.0

    output = forward_model(model, batch, "joint_action_stage")
    changed_output = forward_model(model, changed, "joint_action_stage")

    torch.testing.assert_close(output, changed_output)
