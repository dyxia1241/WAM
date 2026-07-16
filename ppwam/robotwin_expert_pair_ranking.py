from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ppwam.data import collate_batch
from ppwam.joint_flow import (
    JointFlowPreparedWindowDataset,
    joint_flow_runtime_options,
    load_joint_flow_checkpoint,
    score_action,
)
from ppwam.manifest import write_manifest
from ppwam.metrics import tie_aware_ranking
from ppwam.robotwin_variant_audit import VARIANTS
from ppwam.train import batch_to_device, set_seed


@dataclass(frozen=True)
class WindowRecord:
    split: str
    task: str
    variant: str
    episode_id: str
    window_index: int
    dataset_index: int
    phi_t: float
    delta_phi_raw: float


def infer_task_variant(episode_id: str, raw_task_id: Any) -> tuple[str, str]:
    task = str(raw_task_id)
    variant = "expert" if "expert_direct" in episode_id else "unknown"
    for item in VARIANTS:
        if item in episode_id:
            variant = item
            marker = f"_{item}_"
            if marker in episode_id:
                task = episode_id.split(marker, 1)[0]
            break
    if variant == "expert" and "_expert_direct_" in episode_id:
        task = episode_id.split("_expert_direct_", 1)[0]
    return task, variant


def load_split_dataset(config: dict[str, Any], split: str) -> JointFlowPreparedWindowDataset:
    data_cfg = config["data"]
    feature_cfg = config["features"]
    return JointFlowPreparedWindowDataset(
        windows_dir=data_cfg["windows_dir"],
        episodes_dir=data_cfg["episodes_dir"],
        features_dir=data_cfg["features_dir"],
        split=split,
        feature_dim=int(feature_cfg["feature_dim"]),
        norm_stats=data_cfg.get("norm_stats"),
        prompt_features=data_cfg.get("prompt_features"),
        prompt_feature_dim=(
            int(data_cfg["prompt_feature_dim"]) if data_cfg.get("prompt_feature_dim") is not None else None
        ),
        canonical_proprio_dim=(
            int(data_cfg["canonical_proprio_dim"]) if data_cfg.get("canonical_proprio_dim") is not None else None
        ),
        canonical_action_dim=(
            int(data_cfg["canonical_action_dim"]) if data_cfg.get("canonical_action_dim") is not None else None
        ),
        canonical_num_cameras=(
            int(data_cfg["canonical_num_cameras"]) if data_cfg.get("canonical_num_cameras") is not None else None
        ),
    )


def load_records(config: dict[str, Any], splits: list[str]) -> tuple[list[JointFlowPreparedWindowDataset], list[WindowRecord]]:
    datasets: list[JointFlowPreparedWindowDataset] = []
    records: list[WindowRecord] = []
    for split in splits:
        dataset = load_split_dataset(config, split)
        dataset_id = len(datasets)
        datasets.append(dataset)
        for local_index, window_index in enumerate(dataset.indices):
            window = dataset.windows[window_index]
            episode_id = str(window["episode_id"])
            task, variant = infer_task_variant(episode_id, window.get("task_id", "unknown"))
            records.append(
                WindowRecord(
                    split=split,
                    task=task,
                    variant=variant,
                    episode_id=episode_id,
                    window_index=int(window_index),
                    dataset_index=int(dataset_id),
                    phi_t=float(dataset.labels["phi_t"][window_index]),
                    delta_phi_raw=float(dataset.labels["delta_phi_raw"][window_index]),
                )
            )
    return datasets, records


def nearest_expert_pairs(records: list[WindowRecord]) -> list[tuple[int, int]]:
    expert_by_task: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        if record.variant == "expert":
            expert_by_task.setdefault(record.task, []).append(index)
    if not expert_by_task:
        raise ValueError("No expert records found. Episode ids must contain 'expert_direct'.")

    pairs: list[tuple[int, int]] = []
    for index, record in enumerate(records):
        if record.variant == "expert":
            continue
        experts = expert_by_task.get(record.task, [])
        if not experts:
            continue
        expert_index = min(experts, key=lambda item: abs(records[item].phi_t - record.phi_t))
        pairs.append((index, expert_index))
    if not pairs:
        raise ValueError("No perturb->expert pairs were constructed.")
    return pairs


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    perturb_scores = torch.tensor([float(row["perturb_score"]) for row in rows], dtype=torch.float32)
    expert_scores = torch.tensor([float(row["expert_score"]) for row in rows], dtype=torch.float32)
    perturb_delta = np.asarray([float(row["perturb_delta_phi_raw"]) for row in rows], dtype=np.float64)
    expert_delta = np.asarray([float(row["expert_delta_phi_raw"]) for row in rows], dtype=np.float64)
    return {
        "num_pairs": float(len(rows)),
        "expert_gt_delta_mean": float(np.mean(expert_delta)),
        "perturb_gt_delta_mean": float(np.mean(perturb_delta)),
        "gt_delta_margin_mean": float(np.mean(expert_delta - perturb_delta)),
        "gt_pairwise_acc": float(np.mean(expert_delta > perturb_delta)),
        "model_expert_score_mean": float(torch.mean(expert_scores).item()),
        "model_perturb_score_mean": float(torch.mean(perturb_scores).item()),
        "model_margin_mean": float(torch.mean(expert_scores - perturb_scores).item()),
        "model_pairwise_acc": float(tie_aware_ranking(expert_scores, perturb_scores)),
    }


def grouped_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics = {f"overall_{key}": value for key, value in summarize_rows(rows).items()}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["task"]), str(row["variant"])), []).append(row)
    for (task, variant), items in sorted(groups.items()):
        summary = summarize_rows(items)
        prefix = f"{task}_{variant}"
        metrics.update({f"{prefix}_{key}": value for key, value in summary.items()})
    return metrics


@torch.no_grad()
def evaluate_expert_pair_ranking(
    *,
    checkpoint: str | Path,
    config_overrides: dict[str, Any],
    output_dir: str | Path,
    splits: list[str],
    batch_size: int,
    max_pairs: int,
    seed: int,
    context_mode: str = "perturb",
    device_name: str | None = None,
) -> dict[str, float]:
    set_seed(seed)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = load_joint_flow_checkpoint(checkpoint, device)
    config = dict(config)
    data_cfg = dict(config.get("data", {}))
    data_cfg.update(config_overrides.get("data", {}))
    config["data"] = data_cfg
    feature_cfg = dict(config.get("features", {}))
    feature_cfg.update(config_overrides.get("features", {}))
    config["features"] = feature_cfg
    runtime = joint_flow_runtime_options(config)
    model.eval()

    datasets, records = load_records(config, splits)
    pairs = nearest_expert_pairs(records)
    if max_pairs > 0:
        pairs = pairs[: int(max_pairs)]

    rows: list[dict[str, Any]] = []
    for start in range(0, len(pairs), int(batch_size)):
        chunk = pairs[start : start + int(batch_size)]
        context_samples = []
        perturb_actions = []
        expert_actions = []
        for perturb_idx, expert_idx in chunk:
            perturb = records[perturb_idx]
            expert = records[expert_idx]
            perturb_sample = datasets[perturb.dataset_index][
                datasets[perturb.dataset_index].indices.index(perturb.window_index)
            ]
            expert_sample = datasets[expert.dataset_index][
                datasets[expert.dataset_index].indices.index(expert.window_index)
            ]
            context_samples.append(expert_sample if context_mode == "expert" else perturb_sample)
            perturb_actions.append(perturb_sample["action_chunk"])
            expert_actions.append(expert_sample["action_chunk"])

        batch = batch_to_device(collate_batch(context_samples), device)
        perturb_action = torch.stack(perturb_actions, dim=0).to(device)
        expert_action = torch.stack(expert_actions, dim=0).to(device)
        perturb_score = score_action(
            model,
            batch,
            action_chunk=perturb_action,
            denoise_steps=runtime["score_denoise_steps"],
            phi_tokens=runtime["phi_tokens"],
            phi_reduce=runtime["phi_reduce"],
            future_obs_init=runtime["future_obs_init"],
            clamp=bool(runtime.get("score_clamp", True)),
        ).detach().cpu()
        expert_score = score_action(
            model,
            batch,
            action_chunk=expert_action,
            denoise_steps=runtime["score_denoise_steps"],
            phi_tokens=runtime["phi_tokens"],
            phi_reduce=runtime["phi_reduce"],
            future_obs_init=runtime["future_obs_init"],
            clamp=bool(runtime.get("score_clamp", True)),
        ).detach().cpu()
        for local, (perturb_idx, expert_idx) in enumerate(chunk):
            perturb = records[perturb_idx]
            expert = records[expert_idx]
            rows.append(
                {
                    "pair_index": len(rows),
                    "task": perturb.task,
                    "variant": perturb.variant,
                    "split": perturb.split,
                    "perturb_episode_id": perturb.episode_id,
                    "expert_episode_id": expert.episode_id,
                    "perturb_window_index": perturb.window_index,
                    "expert_window_index": expert.window_index,
                    "context_mode": context_mode,
                    "perturb_phi_t": perturb.phi_t,
                    "expert_phi_t": expert.phi_t,
                    "phi_t_abs_gap": abs(perturb.phi_t - expert.phi_t),
                    "perturb_delta_phi_raw": perturb.delta_phi_raw,
                    "expert_delta_phi_raw": expert.delta_phi_raw,
                    "perturb_score": float(perturb_score[local]),
                    "expert_score": float(expert_score[local]),
                    "model_margin": float(expert_score[local] - perturb_score[local]),
                    "model_prefers_expert": float(expert_score[local] > perturb_score[local]),
                }
            )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "expert_pair_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    metrics = grouped_metrics(rows)
    (output_dir / "expert_pair_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest_config = dict(config)
    manifest_config["expert_pair_ranking"] = {
        "checkpoint": str(checkpoint),
        "splits": splits,
        "max_pairs": int(max_pairs),
        "batch_size": int(batch_size),
        "context_mode": context_mode,
        "pairing": "same_task_nearest_phi_t",
    }
    write_manifest(
        output_dir / "manifest.json",
        kind="eval",
        config=manifest_config,
        metrics=metrics,
        experiment="robotwin_expert_pair_ranking",
        checkpoint=str(checkpoint),
        split=",".join(splits),
        repo_root=Path(__file__).resolve().parents[1],
    )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank direct expert actions against RoboTwin perturbed logged actions.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--windows-dir", required=True)
    parser.add_argument("--episodes-dir", required=True)
    parser.add_argument("--features-dir", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--prompt-features", required=True)
    parser.add_argument("--prompt-feature-dim", type=int, default=32)
    parser.add_argument("--feature-dim", type=int, default=768)
    parser.add_argument("--splits", default="train,val,test")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--context-mode", default="perturb", choices=("perturb", "expert"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    splits = [item.strip() for item in str(args.splits).split(",") if item.strip()]
    metrics = evaluate_expert_pair_ranking(
        checkpoint=args.checkpoint,
        config_overrides={
            "data": {
                "windows_dir": args.windows_dir,
                "episodes_dir": args.episodes_dir,
                "features_dir": args.features_dir,
                "norm_stats": args.norm_stats,
                "prompt_features": args.prompt_features,
                "prompt_feature_dim": int(args.prompt_feature_dim),
            },
            "features": {"feature_dim": int(args.feature_dim)},
        },
        output_dir=args.output_dir,
        splits=splits,
        batch_size=int(args.batch_size),
        max_pairs=int(args.max_pairs),
        seed=int(args.seed),
        context_mode=str(args.context_mode),
        device_name=args.device,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
