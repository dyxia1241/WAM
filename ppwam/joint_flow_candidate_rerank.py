from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ppwam.joint_flow import (
    JointFlowDiT,
    joint_flow_runtime_options,
    load_joint_flow_checkpoint,
    make_joint_flow_loaders,
)
from ppwam.joint_flow_rerank import (
    DEFAULT_CANDIDATE_TYPES,
    CandidateBank,
    build_candidate_bank,
    parse_candidate_types,
    select_hard_candidate_indices,
)
from ppwam.joint_flow_sample_select import (
    _action_smoothness,
    build_selector_scores,
    sample_joint_futures,
    score_candidate_actions,
)
from ppwam.manifest import write_manifest
from ppwam.metrics import tie_aware_ranking
from ppwam.train import batch_to_device, set_seed


DEFAULT_PERTURB_SIGMAS = (0.10, 0.25)
DEFAULT_SELECTOR_NAMES = (
    "random_candidate",
    "max_model_phi",
    "max_proposal_phi",
    "calibrated_gap",
    "calibrated_smooth",
)


@dataclass(frozen=True)
class CandidatePool:
    actions: torch.Tensor
    candidate_types: list[str]
    source_indices: np.ndarray
    target_delta_phi: np.ndarray
    proposal_phi: torch.Tensor


def _nearest_by_mask(bank: CandidateBank, anchor_indices: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    anchors = bank.obs_embedding[anchor_indices].astype(np.float32)
    bank_emb = bank.obs_embedding.astype(np.float32)
    distances = (
        np.sum(anchors * anchors, axis=1, keepdims=True)
        + np.sum(bank_emb * bank_emb, axis=1, keepdims=True).T
        - 2.0 * anchors @ bank_emb.T
    )
    fallback = np.ones_like(valid_mask, dtype=bool)
    fallback[np.arange(len(anchor_indices)), anchor_indices] = False
    valid = valid_mask.copy()
    empty = ~np.any(valid, axis=1)
    if np.any(empty):
        valid[empty] = fallback[empty]
    masked = np.where(valid, distances, np.inf)
    selected = np.argmin(masked, axis=1).astype(np.int64)
    if np.any(~np.isfinite(masked[np.arange(len(anchor_indices)), selected])):
        raise ValueError("Failed to select nearest masked candidates.")
    return selected


def select_same_task_stage_nearest(bank: CandidateBank, anchor_indices: np.ndarray) -> np.ndarray:
    anchors = np.asarray(anchor_indices, dtype=np.int64)
    same_task = bank.task_id[None, :] == bank.task_id[anchors, None]
    same_stage = bank.stage_id[None, :] == bank.stage_id[anchors, None]
    not_self = np.ones((len(anchors), len(bank)), dtype=bool)
    not_self[np.arange(len(anchors)), anchors] = False
    return _nearest_by_mask(bank, anchors, same_task & same_stage & not_self)


def _stack_actions(items: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([item.float() for item in items], dim=1)


def _smooth_perturbation(
    action: torch.Tensor,
    sigma: float,
    generator: torch.Generator,
) -> torch.Tensor:
    noise = torch.randn(action.shape, dtype=action.dtype, device=action.device, generator=generator) * float(sigma)
    if noise.shape[1] > 2:
        smoothed = noise.clone()
        smoothed[:, 1:-1] = 0.25 * noise[:, :-2] + 0.5 * noise[:, 1:-1] + 0.25 * noise[:, 2:]
        noise = smoothed
    return action + noise


@torch.no_grad()
def build_candidate_pool(
    *,
    batch: dict[str, torch.Tensor],
    bank: CandidateBank,
    anchor_indices: np.ndarray,
    generator_model: nn.Module,
    generator_runtime: dict[str, Any],
    device: torch.device,
    hard_candidate_types: list[str],
    far_progress_threshold: float,
    num_wam_samples: int,
    perturb_sigmas: list[float],
    seed_generator: torch.Generator,
    sample_denoise_steps: int,
    rescore_batch_size: int,
) -> CandidatePool:
    if int(num_wam_samples) > 0 and not isinstance(generator_model, JointFlowDiT):
        raise ValueError("WAM-sampled candidates require a joint-flow generator checkpoint.")

    batch_size = int(batch["action_chunk"].shape[0])
    candidate_types: list[str] = ["logged"]
    source_indices = [anchor_indices.astype(np.int64)]
    target_delta = [batch["delta_phi"].detach().cpu().numpy().astype(np.float32)]
    action_items = [batch["action_chunk"].float()]

    hard_indices = select_hard_candidate_indices(
        bank,
        anchor_indices,
        candidate_types=hard_candidate_types,
        far_progress_threshold=float(far_progress_threshold),
    )
    for kind in hard_candidate_types:
        indices = hard_indices[kind].astype(np.int64)
        candidate_types.append(kind)
        source_indices.append(indices)
        target_delta.append(bank.delta_phi[indices].astype(np.float32))
        action_items.append(bank.action_chunk[indices].to(device).float())

    same_stage = select_same_task_stage_nearest(bank, anchor_indices)
    candidate_types.append("same_task_stage_nearest")
    source_indices.append(same_stage.astype(np.int64))
    target_delta.append(bank.delta_phi[same_stage].astype(np.float32))
    action_items.append(bank.action_chunk[same_stage].to(device).float())

    for sigma in perturb_sigmas:
        candidate_types.append(f"smooth_perturb_{float(sigma):.2f}")
        source_indices.append(anchor_indices.astype(np.int64))
        target_delta.append(np.full((batch_size,), np.nan, dtype=np.float32))
        action_items.append(_smooth_perturbation(batch["action_chunk"].float(), float(sigma), seed_generator))

    wam_generated_phi: torch.Tensor | None = None
    if int(num_wam_samples) > 0:
        futures = sample_joint_futures(
            generator_model,
            batch,
            num_samples=int(num_wam_samples),
            denoise_steps=int(sample_denoise_steps),
            phi_tokens=int(generator_runtime["phi_tokens"]),
            phi_reduce=str(generator_runtime["phi_reduce"]),
            clamp=bool(generator_runtime.get("score_clamp", True)),
            generator=seed_generator,
        )
        wam_generated_phi = futures["generated_phi"].detach()
        for sample_index in range(int(num_wam_samples)):
            candidate_types.append(f"wam_sample_{sample_index}")
            source_indices.append(np.full((batch_size,), -1, dtype=np.int64))
            target_delta.append(np.full((batch_size,), np.nan, dtype=np.float32))
            action_items.append(futures["action"][:, sample_index].float())

    actions = _stack_actions(action_items)
    proposal_phi = score_candidate_actions(
        generator_model,
        batch,
        actions,
        runtime=generator_runtime,
        chunk_size=int(rescore_batch_size),
    )
    if wam_generated_phi is not None:
        start = len(candidate_types) - int(num_wam_samples)
        proposal_phi[:, start:] = wam_generated_phi

    return CandidatePool(
        actions=actions,
        candidate_types=candidate_types,
        source_indices=np.stack(source_indices, axis=1),
        target_delta_phi=np.stack(target_delta, axis=1),
        proposal_phi=proposal_phi,
    )


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _mean_finite(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if finite.size else 0.0


def _fraction(values: list[str], prefix: str) -> float:
    if not values:
        return 0.0
    return float(np.mean([item.startswith(prefix) for item in values]))


def _pairwise_metrics(
    prefix: str,
    score_matrix: torch.Tensor,
    candidate_types: list[str],
    strict_negative_types: list[str],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if score_matrix.numel() == 0:
        return metrics
    positive = score_matrix[:, 0].detach().cpu()
    negatives: list[torch.Tensor] = []
    type_to_index = {name: index for index, name in enumerate(candidate_types)}
    for kind in strict_negative_types:
        index = type_to_index.get(kind)
        if index is None:
            continue
        negative = score_matrix[:, index].detach().cpu()
        negatives.append(negative)
        metrics[f"{prefix}_{kind}_pairwise_acc"] = float(tie_aware_ranking(positive, negative))
        metrics[f"{prefix}_{kind}_mean_margin"] = float(torch.mean(positive - negative).item())
    if negatives:
        all_neg = torch.cat(negatives, dim=0)
        all_pos = positive.repeat(len(negatives))
        metrics[f"{prefix}_strict_pairwise_acc"] = float(tie_aware_ranking(all_pos, all_neg))
        metrics[f"{prefix}_strict_mean_margin"] = float(torch.mean(all_pos - all_neg).item())
        stacked = torch.stack(negatives, dim=1)
        best_negative = torch.max(stacked, dim=1).values
        metrics[f"{prefix}_strict_top1_logged_rate"] = float(torch.mean((positive > best_negative).float()).item())
        metrics[f"{prefix}_strict_mean_margin_to_best_negative"] = float(torch.mean(positive - best_negative).item())
    return metrics


def _selector_summary(
    selector_name: str,
    selected_indices: torch.Tensor,
    selector_scores: torch.Tensor,
    model_phi: torch.Tensor,
    proposal_phi: torch.Tensor,
    smoothness: torch.Tensor,
    action_mse: torch.Tensor,
    target_delta: np.ndarray,
    candidate_types: list[str],
) -> dict[str, float | list[str]]:
    batch = torch.arange(model_phi.shape[0], device=model_phi.device)
    selected_model = model_phi[batch, selected_indices]
    selected_proposal = proposal_phi[batch, selected_indices]
    selected_smooth = smoothness[batch, selected_indices]
    selected_mse = action_mse[batch, selected_indices]
    selected_score = selector_scores[batch, selected_indices]
    selected_gap = torch.abs(selected_proposal - selected_model)
    logged_model = model_phi[:, 0]
    logged_proposal = proposal_phi[:, 0]
    selected_types = [candidate_types[int(index)] for index in selected_indices.detach().cpu().tolist()]
    selected_delta = [float(target_delta[row, int(index)]) for row, index in enumerate(selected_indices.detach().cpu())]
    return {
        f"{selector_name}_selection_score_mean": float(selected_score.mean().detach().cpu()),
        f"{selector_name}_model_phi_mean": float(selected_model.mean().detach().cpu()),
        f"{selector_name}_proposal_phi_mean": float(selected_proposal.mean().detach().cpu()),
        f"{selector_name}_minus_logged_model_phi": float((selected_model - logged_model).mean().detach().cpu()),
        f"{selector_name}_minus_logged_proposal_phi": float((selected_proposal - logged_proposal).mean().detach().cpu()),
        f"{selector_name}_proposal_model_gap_mean": float(selected_gap.mean().detach().cpu()),
        f"{selector_name}_action_smoothness_mean": float(selected_smooth.mean().detach().cpu()),
        f"{selector_name}_action_mse_to_logged_mean": float(selected_mse.mean().detach().cpu()),
        f"{selector_name}_selected_logged_rate": float(np.mean([item == "logged" for item in selected_types])),
        f"{selector_name}_selected_wam_rate": _fraction(selected_types, "wam_sample_"),
        f"{selector_name}_selected_perturb_rate": _fraction(selected_types, "smooth_perturb_"),
        f"{selector_name}_selected_bank_delta_phi_mean": _mean_finite(selected_delta),
        "_selected_types": selected_types,
    }


@torch.no_grad()
def evaluate_candidate_reranking(
    scorer_model: nn.Module,
    scorer_config: dict[str, Any],
    loader: DataLoader,
    device: torch.device,
    output_dir: str | Path,
    *,
    generator_model: nn.Module | None = None,
    generator_config: dict[str, Any] | None = None,
    hard_candidate_types: list[str] | None = None,
    perturb_sigmas: list[float] | None = None,
    num_wam_samples: int = 4,
    far_progress_threshold: float = 0.35,
    max_anchors: int = 0,
    seed: int = 42,
    split: str = "test",
    rescore_batch_size: int = 512,
    generated_weight: float = 1.0,
    rescore_weight: float = 1.0,
    gap_weight: float = 1.0,
    smoothness_weight: float = 0.1,
) -> dict[str, float]:
    scorer_model.eval()
    if generator_model is None:
        generator_model = scorer_model
    generator_model.eval()
    if generator_config is None:
        generator_config = scorer_config
    hard_types = list(hard_candidate_types or DEFAULT_CANDIDATE_TYPES)
    perturb = [float(item) for item in (perturb_sigmas if perturb_sigmas is not None else DEFAULT_PERTURB_SIGMAS)]
    scorer_runtime = joint_flow_runtime_options(scorer_config)
    generator_runtime = joint_flow_runtime_options(generator_config)
    sample_steps = int(generator_runtime["score_denoise_steps"])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    bank = build_candidate_bank(loader)
    rng = torch.Generator(device=device)
    rng.manual_seed(int(seed) + 20_011)
    random_np = np.random.default_rng(int(seed) + 71)

    candidate_rows: list[dict[str, float | int | str]] = []
    selection_rows: list[dict[str, float | int | str]] = []
    metric_lists: dict[str, list[float]] = {}
    selected_type_lists: dict[str, list[str]] = {name: [] for name in DEFAULT_SELECTOR_NAMES}
    pairwise_scores: dict[str, list[torch.Tensor]] = {
        "model_phi": [],
        "proposal_phi": [],
        "calibrated_gap": [],
        "calibrated_smooth": [],
    }
    final_candidate_types: list[str] | None = None

    offset = 0
    processed = 0
    for raw_batch in loader:
        batch_size = int(raw_batch["action_chunk"].shape[0])
        anchor_indices = np.arange(offset, offset + batch_size, dtype=np.int64)
        offset += batch_size
        if int(max_anchors) > 0:
            remaining = int(max_anchors) - processed
            if remaining <= 0:
                break
            if remaining < batch_size:
                raw_batch = {key: value[:remaining] for key, value in raw_batch.items()}
                anchor_indices = anchor_indices[:remaining]
                batch_size = remaining
        batch = batch_to_device(raw_batch, device)
        pool = build_candidate_pool(
            batch=batch,
            bank=bank,
            anchor_indices=anchor_indices,
            generator_model=generator_model,
            generator_runtime=generator_runtime,
            device=device,
            hard_candidate_types=hard_types,
            far_progress_threshold=float(far_progress_threshold),
            num_wam_samples=int(num_wam_samples),
            perturb_sigmas=perturb,
            seed_generator=rng,
            sample_denoise_steps=sample_steps,
            rescore_batch_size=int(rescore_batch_size),
        )
        final_candidate_types = pool.candidate_types
        model_phi = score_candidate_actions(
            scorer_model,
            batch,
            pool.actions,
            runtime=scorer_runtime,
            chunk_size=int(rescore_batch_size),
        )
        proposal_phi = pool.proposal_phi
        flat_action = pool.actions.reshape(batch_size * pool.actions.shape[1], *pool.actions.shape[2:])
        smoothness = _action_smoothness(flat_action).reshape(batch_size, pool.actions.shape[1])
        action_mse = torch.mean((pool.actions - batch["action_chunk"].unsqueeze(1)) ** 2, dim=(2, 3))
        selector_mats = build_selector_scores(
            proposal_phi,
            model_phi,
            smoothness,
            generated_weight=float(generated_weight),
            rescore_weight=float(rescore_weight),
            gap_weight=float(gap_weight),
            smoothness_weight=float(smoothness_weight),
        )
        selector_mats["max_model_phi"] = model_phi
        selector_mats["max_proposal_phi"] = proposal_phi
        random_indices = torch.as_tensor(
            random_np.integers(0, pool.actions.shape[1], size=batch_size),
            dtype=torch.long,
            device=device,
        )
        selector_indices = {
            "random_candidate": random_indices,
            "max_model_phi": torch.argmax(selector_mats["max_model_phi"], dim=1),
            "max_proposal_phi": torch.argmax(selector_mats["max_proposal_phi"], dim=1),
            "calibrated_gap": torch.argmax(selector_mats["calibrated_gap"], dim=1),
            "calibrated_smooth": torch.argmax(selector_mats["calibrated_smooth"], dim=1),
        }
        selector_score_for_summary = {
            "random_candidate": torch.zeros_like(model_phi),
            "max_model_phi": selector_mats["max_model_phi"],
            "max_proposal_phi": selector_mats["max_proposal_phi"],
            "calibrated_gap": selector_mats["calibrated_gap"],
            "calibrated_smooth": selector_mats["calibrated_smooth"],
        }

        pairwise_scores["model_phi"].append(model_phi.detach().cpu())
        pairwise_scores["proposal_phi"].append(proposal_phi.detach().cpu())
        pairwise_scores["calibrated_gap"].append(selector_mats["calibrated_gap"].detach().cpu())
        pairwise_scores["calibrated_smooth"].append(selector_mats["calibrated_smooth"].detach().cpu())

        for local_index in range(batch_size):
            anchor = int(anchor_indices[local_index])
            for candidate_index, candidate_type in enumerate(pool.candidate_types):
                candidate_rows.append(
                    {
                        "index": anchor,
                        "candidate_index": int(candidate_index),
                        "candidate_type": candidate_type,
                        "source_index": int(pool.source_indices[local_index, candidate_index]),
                        "target_delta_phi": float(pool.target_delta_phi[local_index, candidate_index]),
                        "proposal_phi": float(proposal_phi[local_index, candidate_index].detach().cpu()),
                        "model_phi": float(model_phi[local_index, candidate_index].detach().cpu()),
                        "proposal_model_gap": float(
                            torch.abs(proposal_phi[local_index, candidate_index] - model_phi[local_index, candidate_index])
                            .detach()
                            .cpu()
                        ),
                        "action_smoothness": float(smoothness[local_index, candidate_index].detach().cpu()),
                        "action_mse_to_logged": float(action_mse[local_index, candidate_index].detach().cpu()),
                    }
                )
            for selector_name, indices in selector_indices.items():
                selected_index = int(indices[local_index].detach().cpu())
                selection_rows.append(
                    {
                        "index": anchor,
                        "selector": selector_name,
                        "selected_candidate_index": selected_index,
                        "selected_candidate_type": pool.candidate_types[selected_index],
                        "selected_source_index": int(pool.source_indices[local_index, selected_index]),
                        "selection_score": float(
                            selector_score_for_summary[selector_name][local_index, selected_index].detach().cpu()
                        ),
                        "selected_model_phi": float(model_phi[local_index, selected_index].detach().cpu()),
                        "selected_proposal_phi": float(proposal_phi[local_index, selected_index].detach().cpu()),
                        "logged_model_phi": float(model_phi[local_index, 0].detach().cpu()),
                        "logged_proposal_phi": float(proposal_phi[local_index, 0].detach().cpu()),
                        "selected_target_delta_phi": float(pool.target_delta_phi[local_index, selected_index]),
                        "selected_action_smoothness": float(smoothness[local_index, selected_index].detach().cpu()),
                        "selected_action_mse_to_logged": float(action_mse[local_index, selected_index].detach().cpu()),
                    }
                )

        for selector_name in DEFAULT_SELECTOR_NAMES:
            summary = _selector_summary(
                selector_name,
                selector_indices[selector_name],
                selector_score_for_summary[selector_name],
                model_phi,
                proposal_phi,
                smoothness,
                action_mse,
                pool.target_delta_phi,
                pool.candidate_types,
            )
            selected_type_lists[selector_name].extend(summary.pop("_selected_types"))  # type: ignore[arg-type]
            for key, value in summary.items():
                metric_lists.setdefault(key, []).append(float(value))
        processed += batch_size

    if final_candidate_types is None:
        raise ValueError("No anchors were evaluated.")

    metrics: dict[str, float] = {
        "num_anchors": float(processed),
        "candidate_count": float(len(final_candidate_types)),
        "num_wam_samples": float(num_wam_samples),
        "rescore_batch_size": float(rescore_batch_size),
        "generated_weight": float(generated_weight),
        "rescore_weight": float(rescore_weight),
        "gap_weight": float(gap_weight),
        "smoothness_weight": float(smoothness_weight),
    }
    for key, values in metric_lists.items():
        metrics[key] = _mean(values)
    for selector_name, selected_types in selected_type_lists.items():
        metrics[f"{selector_name}_selected_logged_rate"] = float(np.mean([item == "logged" for item in selected_types]))
        metrics[f"{selector_name}_selected_wam_rate"] = _fraction(selected_types, "wam_sample_")
        metrics[f"{selector_name}_selected_perturb_rate"] = _fraction(selected_types, "smooth_perturb_")
        for candidate_type in final_candidate_types:
            metrics[f"{selector_name}_type_rate/{candidate_type}"] = float(
                np.mean([item == candidate_type for item in selected_types])
            )

    for prefix, chunks in pairwise_scores.items():
        matrix = torch.cat(chunks, dim=0)
        metrics.update(_pairwise_metrics(prefix, matrix, final_candidate_types, hard_types))

    with (output_path / "candidate_rerank_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    if candidate_rows:
        with (output_path / "candidate_rerank_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(candidate_rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(candidate_rows)
    if selection_rows:
        with (output_path / "candidate_rerank_selection.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(selection_rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(selection_rows)
    with (output_path / "candidate_types.json").open("w", encoding="utf-8") as handle:
        json.dump(final_candidate_types, handle, indent=2)
    return metrics


def _parse_perturb_sigmas(value: str) -> list[float]:
    if not value.strip():
        return []
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate mixed semantic/base-policy candidate reranking.")
    parser.add_argument("--checkpoint", required=True, help="Scorer checkpoint.")
    parser.add_argument("--generator-checkpoint", default="", help="Joint-flow generator checkpoint for WAM samples.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default=None)
    parser.add_argument("--candidate-types", default=",".join(DEFAULT_CANDIDATE_TYPES))
    parser.add_argument("--perturb-sigmas", default=",".join(str(item) for item in DEFAULT_PERTURB_SIGMAS))
    parser.add_argument("--num-wam-samples", type=int, default=4)
    parser.add_argument("--far-progress-threshold", type=float, default=0.35)
    parser.add_argument("--max-anchors", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rescore-batch-size", type=int, default=512)
    parser.add_argument("--generated-weight", type=float, default=1.0)
    parser.add_argument("--rescore-weight", type=float, default=1.0)
    parser.add_argument("--gap-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(int(args.seed))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    scorer_model, scorer_config = load_joint_flow_checkpoint(args.checkpoint, device)
    scorer_config = dict(scorer_config)
    scorer_config["_checkpoint"] = str(args.checkpoint)
    generator_model: nn.Module | None = None
    generator_config: dict[str, Any] | None = None
    if args.generator_checkpoint:
        generator_model, generator_config = load_joint_flow_checkpoint(args.generator_checkpoint, device)
        generator_config = dict(generator_config)
        generator_config["_checkpoint"] = str(args.generator_checkpoint)
    elif isinstance(scorer_model, JointFlowDiT):
        generator_model = scorer_model
        generator_config = scorer_config
    elif int(args.num_wam_samples) > 0:
        raise ValueError("--generator-checkpoint is required when scoring with phi-only and --num-wam-samples > 0.")

    loaders = make_joint_flow_loaders(scorer_config)
    checkpoint_name = "candidate_rerank"
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.checkpoint).resolve().parent / f"{checkpoint_name}_{args.split}"
    )
    metrics = evaluate_candidate_reranking(
        scorer_model=scorer_model,
        scorer_config=scorer_config,
        loader=loaders[args.split],
        device=device,
        output_dir=output_dir,
        generator_model=generator_model,
        generator_config=generator_config,
        hard_candidate_types=parse_candidate_types(args.candidate_types),
        perturb_sigmas=_parse_perturb_sigmas(args.perturb_sigmas),
        num_wam_samples=int(args.num_wam_samples),
        far_progress_threshold=float(args.far_progress_threshold),
        max_anchors=int(args.max_anchors),
        seed=int(args.seed),
        split=str(args.split),
        rescore_batch_size=int(args.rescore_batch_size),
        generated_weight=float(args.generated_weight),
        rescore_weight=float(args.rescore_weight),
        gap_weight=float(args.gap_weight),
        smoothness_weight=float(args.smoothness_weight),
    )
    manifest_config = dict(scorer_config)
    manifest_config["candidate_rerank"] = {
        "generator_checkpoint": str(args.generator_checkpoint or args.checkpoint),
        "split": str(args.split),
        "candidate_types": parse_candidate_types(args.candidate_types),
        "perturb_sigmas": _parse_perturb_sigmas(args.perturb_sigmas),
        "num_wam_samples": int(args.num_wam_samples),
        "max_anchors": int(args.max_anchors),
        "seed": int(args.seed),
    }
    write_manifest(
        output_dir / "manifest.json",
        kind="eval",
        config=manifest_config,
        metrics=metrics,
        experiment="mixed_candidate_reranking",
        checkpoint=str(args.checkpoint),
        split=str(args.split),
        repo_root=Path(__file__).resolve().parents[1],
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
