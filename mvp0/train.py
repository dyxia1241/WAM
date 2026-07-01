from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from mvp0.config import apply_overrides, load_config
from mvp0.counterfactual import make_negative_batch
from mvp0.data import MockDatasetConfig, PreparedWindowDataset, collate_batch, make_mock_splits
from mvp0.losses import counterfactual_ranking_loss, delta_phi_loss
from mvp0.manifest import write_manifest
from mvp0.metrics import compute_metrics
from mvp0.model import MLPCritic, PromptFiLMTransformerCritic, StageFiLMTransformerCritic, TimePrior


EXPERIMENTS = {
    "time_prior",
    "obs_stage",
    "obs_action",
    "joint_action_stage",
    "obs_action_stage",
    "obs_action_stage_cf",
    "obs_action_stage_cf_zero",
    "obs_action_stage_cf_multi",
    "obs_prompt",
    "obs_action_prompt",
    "obs_action_prompt_cf_multi",
}

COUNTERFACTUAL_EXPERIMENTS = {
    "obs_action_stage_cf",
    "obs_action_stage_cf_zero",
    "obs_action_stage_cf_multi",
    "obs_action_prompt_cf_multi",
}

PROMPT_EXPERIMENTS = {
    "obs_prompt",
    "obs_action_prompt",
    "obs_action_prompt_cf_multi",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MVP-0 toy training entrypoint.")
    parser.add_argument("--config", default="mvp0/configs/debug.yaml")
    parser.add_argument("overrides", nargs="*", help="Optional key=value overrides.")
    return parser


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def apply_experiment_mask(
    batch: dict[str, torch.Tensor],
    experiment: str,
) -> dict[str, torch.Tensor]:
    masked = {key: value.clone() for key, value in batch.items()}
    if experiment in {"obs_stage"}:
        masked["action_chunk"] = torch.zeros_like(masked["action_chunk"])
    if experiment in {"obs_action"}:
        masked["stage_id"] = torch.zeros_like(masked["stage_id"])
    if experiment in {"joint_action_stage"}:
        masked["obs_features"] = torch.zeros_like(masked["obs_features"])
    return masked


def parse_negative_types(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise ValueError(f"Unsupported negative type config: {value!r}")


def expand_negative_types(negative_cfg: dict[str, Any]) -> list[str]:
    types = parse_negative_types(negative_cfg.get("types", ["zero"]))
    expanded: list[str] = []
    for negative_type in types:
        if negative_type == "scaled":
            for value in negative_cfg.get("scaled_values", [0.25]):
                expanded.append(f"scaled_{value}")
        else:
            expanded.append(negative_type)
    return expanded


def training_negative_types(config: dict[str, Any], experiment: str) -> list[str]:
    if experiment not in COUNTERFACTUAL_EXPERIMENTS:
        return []
    if experiment in {"obs_action_stage_cf", "obs_action_stage_cf_zero"}:
        return [str(config.get("negative_kind", "zero"))]

    negative_cfg = config.get("negatives", {})
    train_types = parse_negative_types(negative_cfg.get("train_types"))
    if train_types:
        return train_types
    return expand_negative_types(negative_cfg)


def build_model(config: dict[str, Any], experiment: str) -> torch.nn.Module:
    data_cfg = config["data"]
    feature_cfg = config["features"]
    model_cfg = config["model"]

    num_tasks = int(data_cfg.get("num_tasks", 2))
    if experiment == "time_prior":
        return TimePrior(num_tasks=num_tasks)

    if experiment in PROMPT_EXPERIMENTS:
        return PromptFiLMTransformerCritic(
            feature_dim=int(feature_cfg["feature_dim"]),
            proprio_dim=int(data_cfg.get("proprio_dim", 14)),
            action_dim=int(data_cfg.get("action_dim", 14)),
            prompt_dim=int(data_cfg.get("prompt_feature_dim", data_cfg.get("prompt_dim", 512))),
            hidden_dim=int(model_cfg["hidden_dim"]),
            transformer_layers=int(model_cfg["transformer_layers"]),
            transformer_heads=int(model_cfg["transformer_heads"]),
            dropout=float(model_cfg.get("dropout", 0.1)),
            use_action=experiment != "obs_prompt",
        )

    if model_cfg.get("name") == "mlp":
        return MLPCritic(
            feature_dim=int(feature_cfg["feature_dim"]),
            proprio_dim=int(data_cfg.get("proprio_dim", 14)),
            action_dim=int(data_cfg.get("action_dim", 14)),
            horizon=int(data_cfg["horizon"]),
            num_tasks=num_tasks,
        )

    return StageFiLMTransformerCritic(
        feature_dim=int(feature_cfg["feature_dim"]),
        proprio_dim=int(data_cfg.get("proprio_dim", 14)),
        action_dim=int(data_cfg.get("action_dim", 14)),
        num_tasks=num_tasks,
        hidden_dim=int(model_cfg["hidden_dim"]),
        transformer_layers=int(model_cfg["transformer_layers"]),
        transformer_heads=int(model_cfg["transformer_heads"]),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )


def forward_model(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    experiment: str,
) -> torch.Tensor:
    if experiment == "time_prior":
        return model(batch["primitive_time"], batch["stage_id"], batch["task_id"])

    if experiment in PROMPT_EXPERIMENTS:
        if "prompt_features" not in batch:
            raise KeyError("prompt experiments require batch['prompt_features'].")
        masked = apply_experiment_mask(batch, experiment)
        return model(
            masked["obs_features"],
            masked["proprio"],
            masked["action_chunk"],
            masked["prompt_features"],
        )

    masked = apply_experiment_mask(batch, experiment)
    return model(
        masked["obs_features"],
        masked["proprio"],
        masked["action_chunk"],
        masked["stage_id"],
        masked["task_id"],
    )


def score_for_checkpoint(metrics: dict[str, float], save_best_by: str) -> float:
    metric_name = save_best_by
    if "/" in metric_name:
        _, metric_name = metric_name.split("/", 1)
    if metric_name not in metrics:
        if metric_name == "ranking_acc" and "delta_phi_mae" in metrics:
            return -float(metrics["delta_phi_mae"])
        raise KeyError(f"Metric {metric_name!r} is not available for checkpoint selection.")

    value = float(metrics[metric_name])
    lower_is_better = (
        metric_name.endswith("_mae")
        or metric_name.endswith("_rmse")
        or metric_name.endswith("_loss")
        or metric_name in {"delta_phi_mae", "delta_phi_rmse", "train_loss"}
    )
    return -value if lower_is_better else value


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    experiment: str,
    device: torch.device,
    negative_kind: str = "zero",
) -> dict[str, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    pos_preds: list[torch.Tensor] = []
    neg_preds: list[torch.Tensor] = []

    for batch in loader:
        batch = batch_to_device(batch, device)
        logits = forward_model(model, batch, experiment)
        preds.append(torch.sigmoid(logits).cpu().reshape(-1))
        targets.append(batch["delta_phi"].cpu().reshape(-1))

        if experiment != "time_prior":
            paired = make_negative_batch(batch, kind=negative_kind)
            pos_logit = forward_model(model, paired.positive, experiment)
            neg_logit = forward_model(model, paired.negative, experiment)
            pos_preds.append(torch.sigmoid(pos_logit).cpu().reshape(-1))
            neg_preds.append(torch.sigmoid(neg_logit).cpu().reshape(-1))

    pred = torch.cat(preds)
    target = torch.cat(targets)
    if pos_preds and neg_preds:
        return compute_metrics(pred, target, torch.cat(pos_preds), torch.cat(neg_preds))
    return compute_metrics(pred, target)


def make_loaders(config: dict[str, Any]) -> dict[str, DataLoader]:
    data_cfg = config["data"]
    feature_cfg = config["features"]
    batch_size = int(data_cfg.get("batch_size", 8))

    if data_cfg.get("windows_dir"):
        windows_dir = data_cfg["windows_dir"]
        episodes_dir = data_cfg["episodes_dir"]
        features_dir = data_cfg["features_dir"]
        return {
            split: DataLoader(
                PreparedWindowDataset(
                    windows_dir=windows_dir,
                    episodes_dir=episodes_dir,
                    features_dir=features_dir,
                    split=split,
                    feature_dim=int(feature_cfg["feature_dim"]),
                    norm_stats=data_cfg.get("norm_stats"),
                    prompt_features=data_cfg.get("prompt_features"),
                    prompt_feature_dim=(
                        int(data_cfg["prompt_feature_dim"])
                        if data_cfg.get("prompt_feature_dim") is not None
                        else None
                    ),
                ),
                batch_size=batch_size,
                shuffle=(split == "train"),
                collate_fn=collate_batch,
            )
            for split in ("train", "val", "test")
        }

    dataset_cfg = MockDatasetConfig(
        num_samples=int(data_cfg.get("num_samples", 64)),
        history=int(data_cfg["history"]),
        horizon=int(data_cfg["horizon"]),
        cameras=int(data_cfg.get("cameras", 1)),
        feature_dim=int(feature_cfg["feature_dim"]),
        proprio_dim=int(data_cfg.get("proprio_dim", 14)),
        action_dim=int(data_cfg.get("action_dim", 14)),
        prompt_dim=int(data_cfg.get("prompt_feature_dim", data_cfg.get("prompt_dim", 512))),
        num_tasks=int(data_cfg.get("num_tasks", 2)),
        seed=int(config.get("seed", 42)),
    )
    datasets = make_mock_splits(dataset_cfg)
    return {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_batch,
        )
        for split, dataset in datasets.items()
    }


def train(config: dict[str, Any]) -> dict[str, float]:
    experiment = config.get("experiment", "obs_action_stage_cf")
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")

    set_seed(int(config.get("seed", 42)))
    device = torch.device(config.get("device", "cpu"))
    output_dir = Path(config.get("output_dir", "outputs")) / experiment
    output_dir.mkdir(parents=True, exist_ok=True)

    loaders = make_loaders(config)
    model = build_model(config, experiment).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["optim"]["lr"]),
        weight_decay=float(config["optim"].get("weight_decay", 0.0)),
    )
    max_epochs = int(config["train"].get("max_epochs", 2))
    delta_weight = float(config["loss"].get("delta_weight", 1.0))
    cf_weight = float(config["loss"].get("counterfactual_weight", 0.5))
    margin = float(config["loss"].get("margin", 0.05))
    save_best_by = str(config.get("train", {}).get("save_best_by", "val/ranking_acc"))
    negative_types = training_negative_types(config, experiment)
    eval_negative_kind = str(config.get("eval_negative_kind", config.get("negative_kind", "zero")))
    rng = np.random.default_rng(int(config.get("seed", 42)))

    best_metric = -float("inf")
    best_metrics: dict[str, float] = {}
    for epoch in range(max_epochs):
        model.train()
        train_losses: list[float] = []
        for batch in loaders["train"]:
            batch = batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_model(model, batch, experiment)
            loss = delta_weight * delta_phi_loss(logits, batch["delta_phi"])

            if experiment in COUNTERFACTUAL_EXPERIMENTS:
                negative_kind = str(negative_types[int(rng.integers(len(negative_types)))])
                paired = make_negative_batch(batch, kind=negative_kind)
                pos_logit = forward_model(model, paired.positive, experiment)
                neg_logit = forward_model(model, paired.negative, experiment)
                loss = loss + cf_weight * counterfactual_ranking_loss(pos_logit, neg_logit, margin=margin)

            loss.backward()
            grad_clip = float(config["optim"].get("grad_clip_norm", 0.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_metrics = evaluate_model(model, loaders["val"], experiment, device, negative_kind=eval_negative_kind)
        val_metrics["train_loss"] = float(np.mean(train_losses))
        val_metrics["epoch"] = float(epoch)
        score = score_for_checkpoint(val_metrics, save_best_by)
        if score > best_metric:
            best_metric = score
            best_metrics = dict(val_metrics)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "experiment": experiment,
                    "metrics": best_metrics,
                },
                output_dir / "best.pt",
            )

    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(best_metrics, handle, indent=2, sort_keys=True)
    write_manifest(
        output_dir / "manifest.json",
        kind="train",
        config=config,
        metrics=best_metrics,
        experiment=experiment,
        checkpoint=str(output_dir / "best.pt"),
        repo_root=Path(__file__).resolve().parents[1],
    )
    print(json.dumps(best_metrics, indent=2, sort_keys=True))
    return best_metrics


def main() -> None:
    args = build_parser().parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    train(config)


if __name__ == "__main__":
    main()
