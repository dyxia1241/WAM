from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ppwam.joint_flow import (
    JointFlowDiT,
    joint_flow_runtime_options,
    load_joint_flow_checkpoint,
    make_joint_flow_loaders,
    score_action,
)
from ppwam.manifest import write_manifest
from ppwam.metrics import tie_aware_ranking
from ppwam.train import batch_to_device


DEFAULT_CANDIDATE_TYPES = (
    "same_task_phase_wrong",
    "same_task_far_progress",
    "cross_task",
    "nearest_obs_wrong_action",
)


@dataclass(frozen=True)
class CandidateBank:
    action_chunk: torch.Tensor
    task_id: np.ndarray
    stage_id: np.ndarray
    primitive_time: np.ndarray
    delta_phi: np.ndarray
    obs_embedding: np.ndarray

    def __len__(self) -> int:
        return int(self.action_chunk.shape[0])


def _pool_history_embedding(obs_features: torch.Tensor) -> torch.Tensor:
    pooled = JointFlowDiT._pool_features(obs_features).float()
    return pooled.mean(dim=1)


def build_candidate_bank(loader: DataLoader) -> CandidateBank:
    actions: list[torch.Tensor] = []
    task_ids: list[np.ndarray] = []
    stage_ids: list[np.ndarray] = []
    primitive_times: list[np.ndarray] = []
    delta_phis: list[np.ndarray] = []
    obs_embeddings: list[np.ndarray] = []

    for batch in loader:
        actions.append(batch["action_chunk"].detach().cpu().float())
        task_ids.append(batch["task_id"].detach().cpu().numpy().astype(np.int64))
        stage_ids.append(batch["stage_id"].detach().cpu().numpy().astype(np.int64))
        primitive_times.append(batch["primitive_time"].detach().cpu().numpy().astype(np.float32))
        delta_phis.append(batch["delta_phi"].detach().cpu().numpy().astype(np.float32))
        obs_embeddings.append(_pool_history_embedding(batch["obs_features"]).detach().cpu().numpy().astype(np.float32))

    if not actions:
        raise ValueError("Cannot build a candidate bank from an empty loader.")
    return CandidateBank(
        action_chunk=torch.cat(actions, dim=0),
        task_id=np.concatenate(task_ids, axis=0),
        stage_id=np.concatenate(stage_ids, axis=0),
        primitive_time=np.concatenate(primitive_times, axis=0),
        delta_phi=np.concatenate(delta_phis, axis=0),
        obs_embedding=np.concatenate(obs_embeddings, axis=0),
    )


def parse_candidate_types(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if value is None:
        return list(DEFAULT_CANDIDATE_TYPES)
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item) for item in value]


def _candidate_mask(
    bank: CandidateBank,
    anchor_indices: np.ndarray,
    kind: str,
    far_progress_threshold: float,
) -> np.ndarray:
    anchor_task = bank.task_id[anchor_indices]
    anchor_stage = bank.stage_id[anchor_indices]
    anchor_time = bank.primitive_time[anchor_indices]
    not_self = np.ones((len(anchor_indices), len(bank)), dtype=bool)
    not_self[np.arange(len(anchor_indices)), anchor_indices] = False

    same_task = bank.task_id[None, :] == anchor_task[:, None]
    diff_task = ~same_task
    diff_stage = bank.stage_id[None, :] != anchor_stage[:, None]
    far_progress = np.abs(bank.primitive_time[None, :] - anchor_time[:, None]) >= far_progress_threshold

    if kind == "same_task_phase_wrong":
        return same_task & diff_stage & not_self
    if kind == "same_task_far_progress":
        return same_task & far_progress & not_self
    if kind == "cross_task":
        return diff_task & not_self
    if kind == "nearest_obs_wrong_action":
        return (diff_task | diff_stage) & not_self
    raise ValueError(f"Unknown hard reranking candidate type: {kind}")


def _nearest_by_observation(
    bank: CandidateBank,
    anchor_indices: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
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
        raise ValueError("Failed to select hard reranking candidates.")
    return selected


def _with_fallback(valid_mask: np.ndarray, anchor_indices: np.ndarray) -> np.ndarray:
    fallback = np.ones_like(valid_mask, dtype=bool)
    fallback[np.arange(len(anchor_indices)), anchor_indices] = False
    valid = valid_mask.copy()
    empty = ~np.any(valid, axis=1)
    if np.any(empty):
        valid[empty] = fallback[empty]
    return valid


def _select_min_score(score: np.ndarray, valid_mask: np.ndarray, anchor_indices: np.ndarray) -> np.ndarray:
    valid = _with_fallback(valid_mask, anchor_indices)
    masked = np.where(valid, score, np.inf)
    selected = np.argmin(masked, axis=1).astype(np.int64)
    if np.any(~np.isfinite(masked[np.arange(len(anchor_indices)), selected])):
        raise ValueError("Failed to select hard reranking candidates.")
    return selected


def _select_max_score(score: np.ndarray, valid_mask: np.ndarray, anchor_indices: np.ndarray) -> np.ndarray:
    valid = _with_fallback(valid_mask, anchor_indices)
    masked = np.where(valid, score, -np.inf)
    selected = np.argmax(masked, axis=1).astype(np.int64)
    if np.any(~np.isfinite(masked[np.arange(len(anchor_indices)), selected])):
        raise ValueError("Failed to select hard reranking candidates.")
    return selected


def select_hard_candidate_indices(
    bank: CandidateBank,
    anchor_indices: np.ndarray,
    candidate_types: list[str] | tuple[str, ...] = DEFAULT_CANDIDATE_TYPES,
    far_progress_threshold: float = 0.35,
) -> dict[str, np.ndarray]:
    anchors = np.asarray(anchor_indices, dtype=np.int64)
    if anchors.ndim != 1:
        raise ValueError("anchor_indices must be a 1-D array.")
    if np.any(anchors < 0) or np.any(anchors >= len(bank)):
        raise ValueError("anchor_indices contains out-of-range values.")

    selected: dict[str, np.ndarray] = {}
    time_distance = np.abs(bank.primitive_time[None, :] - bank.primitive_time[anchors, None])
    stage_mismatch = (bank.stage_id[None, :] != bank.stage_id[anchors, None]).astype(np.float32)
    for kind in candidate_types:
        mask = _candidate_mask(bank, anchors, kind, far_progress_threshold=far_progress_threshold)
        if kind == "same_task_phase_wrong":
            selected[kind] = _select_min_score(time_distance, mask, anchors)
        elif kind == "same_task_far_progress":
            selected[kind] = _select_max_score(time_distance, mask, anchors)
        elif kind == "cross_task":
            selected[kind] = _select_min_score(time_distance + 0.25 * stage_mismatch, mask, anchors)
        elif kind == "nearest_obs_wrong_action":
            selected[kind] = _nearest_by_observation(bank, anchors, mask)
        else:
            raise ValueError(f"Unknown hard reranking candidate type: {kind}")
    return selected


def reranking_metrics(rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    if not rows:
        return {}
    pos = torch.tensor([float(row["positive_score"]) for row in rows], dtype=torch.float32)
    neg = torch.tensor([float(row["candidate_score"]) for row in rows], dtype=torch.float32)
    metrics = {
        "hard_pairwise_ranking_acc": float(tie_aware_ranking(pos, neg)),
        "hard_pairwise_mean_margin": float(torch.mean(pos - neg).item()),
    }
    by_type: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        by_type.setdefault(str(row["candidate_type"]), []).append(index)
    for kind, indices in sorted(by_type.items()):
        kind_pos = pos[indices]
        kind_neg = neg[indices]
        metrics[f"{kind}_ranking_acc"] = float(tie_aware_ranking(kind_pos, kind_neg))
        metrics[f"{kind}_mean_margin"] = float(torch.mean(kind_pos - kind_neg).item())
    return metrics


def top1_metrics(selection_rows: list[dict[str, float | int | str]]) -> dict[str, float]:
    if not selection_rows:
        return {}
    return {
        "hard_top1_acc": float(np.mean([float(row["strict_top1"]) for row in selection_rows])),
        "hard_tie_aware_top1_acc": float(np.mean([float(row["tie_aware_top1_credit"]) for row in selection_rows])),
        "hard_top3_acc": float(np.mean([float(row["positive_rank"]) <= 3.0 for row in selection_rows])),
        "hard_mean_positive_rank": float(np.mean([float(row["positive_rank"]) for row in selection_rows])),
        "hard_mean_margin_to_best_negative": float(
            np.mean([float(row["positive_score"]) - float(row["best_negative_score"]) for row in selection_rows])
        ),
        "hard_candidate_count_mean": float(np.mean([float(row["candidate_count"]) for row in selection_rows])),
    }


@torch.no_grad()
def evaluate_hard_reranking(
    model: torch.nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    output_dir: str | Path | None = None,
    candidate_types: list[str] | tuple[str, ...] = DEFAULT_CANDIDATE_TYPES,
    far_progress_threshold: float = 0.35,
    max_anchors: int | None = None,
) -> dict[str, float]:
    model.eval()
    runtime = joint_flow_runtime_options(config)
    bank = build_candidate_bank(loader)
    rows: list[dict[str, float | int | str]] = []
    selection_rows: list[dict[str, float | int | str]] = []
    offset = 0

    for batch in loader:
        batch_size = int(batch["action_chunk"].shape[0])
        anchor_indices = np.arange(offset, offset + batch_size, dtype=np.int64)
        if max_anchors is not None:
            remaining = max(0, int(max_anchors) - len(selection_rows))
            if remaining <= 0:
                break
            if remaining < batch_size:
                batch = {key: value[:remaining] for key, value in batch.items()}
                anchor_indices = anchor_indices[:remaining]
                batch_size = remaining

        selected = select_hard_candidate_indices(
            bank,
            anchor_indices,
            candidate_types=candidate_types,
            far_progress_threshold=far_progress_threshold,
        )
        device_batch = batch_to_device(batch, device)
        positive_scores = score_action(
            model,
            device_batch,
            action_chunk=device_batch["action_chunk"],
            clamp=True,
            denoise_steps=runtime["score_denoise_steps"],
            phi_tokens=runtime["phi_tokens"],
            phi_reduce=runtime["phi_reduce"],
            future_obs_init=runtime["future_obs_init"],
        ).detach().cpu()

        candidate_scores_by_type: dict[str, torch.Tensor] = {}
        for kind, source_indices in selected.items():
            candidate_action = bank.action_chunk[source_indices].to(device)
            candidate_scores_by_type[kind] = score_action(
                model,
                device_batch,
                action_chunk=candidate_action,
                clamp=True,
                denoise_steps=runtime["score_denoise_steps"],
                phi_tokens=runtime["phi_tokens"],
                phi_reduce=runtime["phi_reduce"],
                future_obs_init=runtime["future_obs_init"],
            ).detach().cpu()

        for local_index in range(batch_size):
            positive_score = float(positive_scores[local_index])
            neg_records: list[tuple[str, int, float]] = []
            for kind in candidate_types:
                source_index = int(selected[kind][local_index])
                candidate_score = float(candidate_scores_by_type[kind][local_index])
                neg_records.append((kind, source_index, candidate_score))
                margin = positive_score - candidate_score
                rows.append(
                    {
                        "index": int(anchor_indices[local_index]),
                        "candidate_type": kind,
                        "source_index": source_index,
                        "positive_score": positive_score,
                        "candidate_score": candidate_score,
                        "margin": margin,
                        "pairwise_correct": int(positive_score > candidate_score),
                    }
                )

            negative_scores = np.asarray([record[2] for record in neg_records], dtype=np.float64)
            best_negative_position = int(np.argmax(negative_scores))
            best_negative_type, best_source_index, best_negative_score = neg_records[best_negative_position]
            score_vector = np.concatenate([np.asarray([positive_score], dtype=np.float64), negative_scores])
            max_score = float(np.max(score_vector))
            tied_best = int(np.sum(np.isclose(score_vector, max_score)))
            positive_is_tied_best = bool(np.isclose(positive_score, max_score))
            tie_credit = (1.0 / tied_best) if positive_is_tied_best else 0.0
            positive_rank = 1 + int(np.sum(negative_scores > positive_score))
            selection_rows.append(
                {
                    "index": int(anchor_indices[local_index]),
                    "positive_score": positive_score,
                    "best_negative_score": float(best_negative_score),
                    "best_negative_type": best_negative_type,
                    "best_negative_source_index": int(best_source_index),
                    "positive_rank": int(positive_rank),
                    "strict_top1": int(positive_score > float(best_negative_score)),
                    "tie_aware_top1_credit": float(tie_credit),
                    "candidate_count": int(len(neg_records) + 1),
                }
            )
        offset += int(batch["action_chunk"].shape[0])

    metrics = reranking_metrics(rows)
    metrics.update(top1_metrics(selection_rows))
    metrics["hard_num_anchors"] = float(len(selection_rows))
    metrics["hard_num_pairwise_candidates"] = float(len(rows))

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "hard_reranking_pairs.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "index",
                    "candidate_type",
                    "source_index",
                    "positive_score",
                    "candidate_score",
                    "margin",
                    "pairwise_correct",
                ],
            )
            writer.writeheader()
            writer.writerows(rows)
        with (out / "hard_reranking_selection.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "index",
                    "positive_score",
                    "best_negative_score",
                    "best_negative_type",
                    "best_negative_source_index",
                    "positive_rank",
                    "strict_top1",
                    "tie_aware_top1_credit",
                    "candidate_count",
                ],
            )
            writer.writeheader()
            writer.writerows(selection_rows)
        with (out / "hard_reranking_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate hard candidate reranking for PP-WAM joint-flow checkpoints.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--candidate-types", default=",".join(DEFAULT_CANDIDATE_TYPES))
    parser.add_argument("--far-progress-threshold", type=float, default=0.35)
    parser.add_argument("--max-anchors", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config = load_joint_flow_checkpoint(args.checkpoint, device)
    config = dict(config)
    config["device"] = str(device)
    loaders = make_joint_flow_loaders(config)
    output = Path(args.output) if args.output is not None else Path(args.checkpoint).parent / f"hard_rerank_{args.split}"
    metrics = evaluate_hard_reranking(
        model=model,
        loader=loaders[args.split],
        config=config,
        device=device,
        output_dir=output,
        candidate_types=parse_candidate_types(args.candidate_types),
        far_progress_threshold=float(args.far_progress_threshold),
        max_anchors=args.max_anchors,
    )
    write_manifest(
        output / "manifest.json",
        kind="eval",
        config=config,
        metrics=metrics,
        experiment="hard_candidate_reranking",
        checkpoint=str(args.checkpoint),
        split=args.split,
        repo_root=Path(__file__).resolve().parents[1],
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
