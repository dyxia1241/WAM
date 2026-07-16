from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ppwam.joint_flow import (
    JointFlowDiT,
    PhiOnlyFlowCritic,
    joint_flow_runtime_options,
    load_joint_flow_checkpoint,
    make_joint_flow_loaders,
    score_action,
)
from ppwam.train import batch_to_device, set_seed


def _randn_like(values: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.randn(values.shape, dtype=values.dtype, device=values.device, generator=generator)


def _phi_score(phi_state: torch.Tensor, reduce: str = "last", clamp: bool = True) -> torch.Tensor:
    if reduce == "last":
        score = phi_state[:, -1]
    elif reduce == "mean":
        score = phi_state.mean(dim=1)
    else:
        raise ValueError(f"Unknown phi reduce mode: {reduce}")
    return score.clamp(0.0, 1.0) if clamp else score


def repeat_batch(batch: dict[str, torch.Tensor], repeats: int) -> dict[str, torch.Tensor]:
    return {key: value.repeat_interleave(repeats, dim=0) for key, value in batch.items()}


@torch.no_grad()
def sample_joint_futures(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    num_samples: int = 16,
    denoise_steps: int = 4,
    phi_tokens: int = 8,
    phi_reduce: str = "last",
    clamp: bool = True,
    generator: torch.Generator | None = None,
) -> dict[str, torch.Tensor]:
    if isinstance(model, PhiOnlyFlowCritic):
        raise ValueError("Imagined-future sampling requires a joint-flow checkpoint, not phi-only.")
    if not isinstance(model, JointFlowDiT):
        raise TypeError(f"Unsupported model type: {type(model).__name__}")

    batch_size = int(batch["action_chunk"].shape[0])
    horizon = int(batch["action_chunk"].shape[1])
    action_dim = int(batch["action_chunk"].shape[2])
    expanded = repeat_batch(batch, int(num_samples))
    future_obs_shape = JointFlowDiT._pool_features(expanded["future_obs_features"]).shape
    action_shape = (batch_size * int(num_samples), horizon, action_dim)
    phi_shape = (batch_size * int(num_samples), max(1, int(phi_tokens)))

    future_obs_state = torch.randn(
        future_obs_shape,
        dtype=expanded["action_chunk"].dtype,
        device=expanded["action_chunk"].device,
        generator=generator,
    )
    action_state = torch.randn(
        action_shape,
        dtype=expanded["action_chunk"].dtype,
        device=expanded["action_chunk"].device,
        generator=generator,
    )
    phi_state = torch.randn(
        phi_shape,
        dtype=expanded["action_chunk"].dtype,
        device=expanded["action_chunk"].device,
        generator=generator,
    )

    steps = max(1, int(denoise_steps))
    dt = 1.0 / float(steps)
    for step in range(steps):
        tau = torch.full(
            (batch_size * int(num_samples),),
            step * dt,
            device=action_state.device,
            dtype=action_state.dtype,
        )
        outputs = model(
            expanded["obs_features"],
            expanded["proprio_history"],
            expanded["prompt_features"],
            future_obs_state,
            action_state,
            phi_state,
            tau,
            action_is_condition=False,
        )
        future_obs_state = future_obs_state + dt * outputs["v_obs"]
        action_state = action_state + dt * outputs["v_action"]
        phi_state = phi_state + dt * outputs["v_phi"]

    scores = _phi_score(phi_state, reduce=phi_reduce, clamp=clamp)
    return {
        "action": action_state.reshape(batch_size, int(num_samples), horizon, action_dim),
        "future_obs": future_obs_state.reshape(batch_size, int(num_samples), *future_obs_state.shape[1:]),
        "phi_state": phi_state.reshape(batch_size, int(num_samples), phi_state.shape[-1]),
        "generated_phi": scores.reshape(batch_size, int(num_samples)),
    }


def _gather_candidates(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch, indices]


def _gather_scores(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch, indices]


def _action_smoothness(action: torch.Tensor) -> torch.Tensor:
    if action.shape[1] <= 1:
        return torch.zeros((action.shape[0],), dtype=action.dtype, device=action.device)
    return torch.mean((action[:, 1:] - action[:, :-1]) ** 2, dim=(1, 2))


def _zscore_candidates(values: torch.Tensor) -> torch.Tensor:
    centered = values - values.mean(dim=1, keepdim=True)
    scale = values.std(dim=1, unbiased=False, keepdim=True).clamp_min(1.0e-6)
    return centered / scale


def _pearson_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2 or right.size < 2:
        return 0.0
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std <= 1.0e-12 or right_std <= 1.0e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


@torch.no_grad()
def score_candidate_actions(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    actions: torch.Tensor,
    runtime: dict[str, Any],
    chunk_size: int = 512,
) -> torch.Tensor:
    batch_size = int(actions.shape[0])
    num_samples = int(actions.shape[1])
    flat_actions = actions.reshape(batch_size * num_samples, *actions.shape[2:])
    expanded = repeat_batch(batch, num_samples)
    scores: list[torch.Tensor] = []
    step = int(chunk_size) if int(chunk_size) > 0 else int(flat_actions.shape[0])
    for start in range(0, int(flat_actions.shape[0]), step):
        end = min(start + step, int(flat_actions.shape[0]))
        chunk = {key: value[start:end] for key, value in expanded.items()}
        scores.append(
            score_action(
                model,
                chunk,
                action_chunk=flat_actions[start:end],
                denoise_steps=runtime["score_denoise_steps"],
                phi_tokens=runtime["phi_tokens"],
                phi_reduce=runtime["phi_reduce"],
                future_obs_init=runtime["future_obs_init"],
                clamp=bool(runtime.get("score_clamp", True)),
            )
        )
    return torch.cat(scores, dim=0).reshape(batch_size, num_samples)


def build_selector_scores(
    generated_phi: torch.Tensor,
    rescored_phi: torch.Tensor,
    smoothness: torch.Tensor,
    generated_weight: float = 1.0,
    rescore_weight: float = 1.0,
    gap_weight: float = 1.0,
    smoothness_weight: float = 0.1,
) -> dict[str, torch.Tensor]:
    gap = torch.abs(generated_phi - rescored_phi)
    calibrated_gap = (
        float(generated_weight) * _zscore_candidates(generated_phi)
        + float(rescore_weight) * _zscore_candidates(rescored_phi)
        - float(gap_weight) * _zscore_candidates(gap)
    )
    calibrated_smooth = calibrated_gap - float(smoothness_weight) * _zscore_candidates(smoothness)
    return {
        "max_generated_phi": generated_phi,
        "max_rescored_phi": rescored_phi,
        "calibrated_gap": calibrated_gap,
        "calibrated_smooth": calibrated_smooth,
    }


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


@torch.no_grad()
def evaluate_sample_selection(
    model: nn.Module,
    loader: DataLoader,
    config: dict[str, Any],
    device: torch.device,
    output_dir: str | Path,
    num_samples: int = 16,
    denoise_steps: int | None = None,
    max_batches: int = 0,
    seed: int = 42,
    split: str = "test",
    rescore_batch_size: int = 512,
    generated_weight: float = 1.0,
    rescore_weight: float = 1.0,
    gap_weight: float = 1.0,
    smoothness_weight: float = 0.1,
) -> dict[str, float]:
    runtime = joint_flow_runtime_options(config)
    sample_steps = int(denoise_steps if denoise_steps is not None else runtime["score_denoise_steps"])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 10_003)
    model.eval()

    rows: list[dict[str, float | int]] = []
    selector_names = ["random_sample", "max_generated_phi", "max_rescored_phi", "calibrated_gap", "calibrated_smooth"]
    selector_stats: dict[str, dict[str, list[float]]] = {
        name: {
            "selection_score": [],
            "generated_phi": [],
            "rescored_phi": [],
            "action_mse_to_logged": [],
            "action_smoothness": [],
            "generation_rescore_gap": [],
        }
        for name in selector_names
    }
    candidate_generated: list[float] = []
    candidate_rescore: list[float] = []
    candidate_gap: list[float] = []
    candidate_smooth: list[float] = []
    logged_rescore: list[float] = []
    target_delta: list[float] = []

    global_index = 0
    for batch_index, batch in enumerate(loader):
        if max_batches > 0 and batch_index >= max_batches:
            break
        batch = batch_to_device(batch, device)
        futures = sample_joint_futures(
            model,
            batch,
            num_samples=num_samples,
            denoise_steps=sample_steps,
            phi_tokens=runtime["phi_tokens"],
            phi_reduce=runtime["phi_reduce"],
            clamp=bool(runtime.get("score_clamp", True)),
            generator=generator,
        )
        generated_phi = futures["generated_phi"]
        candidate_scores = score_candidate_actions(
            model,
            batch,
            futures["action"],
            runtime=runtime,
            chunk_size=int(rescore_batch_size),
        )
        logged_score = score_action(
            model,
            batch,
            action_chunk=batch["action_chunk"],
            denoise_steps=runtime["score_denoise_steps"],
            phi_tokens=runtime["phi_tokens"],
            phi_reduce=runtime["phi_reduce"],
            future_obs_init=runtime["future_obs_init"],
            clamp=bool(runtime.get("score_clamp", True)),
        )

        batch_size = int(batch["action_chunk"].shape[0])
        flat_action = futures["action"].reshape(batch_size * int(num_samples), *futures["action"].shape[2:])
        smoothness = _action_smoothness(flat_action).reshape(batch_size, int(num_samples))
        action_mse = torch.mean(
            (futures["action"] - batch["action_chunk"].unsqueeze(1)) ** 2,
            dim=(2, 3),
        )
        gap = torch.abs(generated_phi - candidate_scores)
        selector_score_matrices = build_selector_scores(
            generated_phi,
            candidate_scores,
            smoothness,
            generated_weight=float(generated_weight),
            rescore_weight=float(rescore_weight),
            gap_weight=float(gap_weight),
            smoothness_weight=float(smoothness_weight),
        )
        selector_score_matrices = {"random_sample": torch.zeros_like(generated_phi), **selector_score_matrices}
        selector_indices = {
            name: torch.zeros((batch_size,), dtype=torch.long, device=generated_phi.device)
            if name == "random_sample"
            else torch.argmax(scores, dim=1)
            for name, scores in selector_score_matrices.items()
        }

        selected_by_name: dict[str, dict[str, torch.Tensor]] = {}
        for name in selector_names:
            indices = selector_indices[name]
            selected_by_name[name] = {
                "indices": indices,
                "selection_score": _gather_scores(selector_score_matrices[name], indices),
                "generated_phi": _gather_scores(generated_phi, indices),
                "rescored_phi": _gather_scores(candidate_scores, indices),
                "action_mse_to_logged": _gather_scores(action_mse, indices),
                "action_smoothness": _gather_scores(smoothness, indices),
                "generation_rescore_gap": _gather_scores(gap, indices),
            }

        for local_index in range(int(batch["action_chunk"].shape[0])):
            row = {
                "index": global_index,
                "target_delta_phi": float(batch["delta_phi"][local_index].detach().cpu()),
                "logged_rescored_phi": float(logged_score[local_index].detach().cpu()),
            }
            for name in selector_names:
                values = selected_by_name[name]
                row[f"{name}_sample"] = int(values["indices"][local_index].detach().cpu())
                row[f"{name}_selection_score"] = float(values["selection_score"][local_index].detach().cpu())
                row[f"{name}_generated_phi"] = float(values["generated_phi"][local_index].detach().cpu())
                row[f"{name}_rescored_phi"] = float(values["rescored_phi"][local_index].detach().cpu())
                row[f"{name}_action_mse_to_logged"] = float(
                    values["action_mse_to_logged"][local_index].detach().cpu()
                )
                row[f"{name}_action_smoothness"] = float(values["action_smoothness"][local_index].detach().cpu())
                row[f"{name}_generation_rescore_gap"] = float(
                    values["generation_rescore_gap"][local_index].detach().cpu()
                )
            rows.append(row)
            global_index += 1

        for name in selector_names:
            for key, values in selector_stats[name].items():
                values.extend(float(item) for item in selected_by_name[name][key].detach().cpu().tolist())
        candidate_generated.extend(float(item) for item in generated_phi.reshape(-1).detach().cpu().tolist())
        candidate_rescore.extend(float(item) for item in candidate_scores.reshape(-1).detach().cpu().tolist())
        candidate_gap.extend(float(item) for item in gap.reshape(-1).detach().cpu().tolist())
        candidate_smooth.extend(float(item) for item in smoothness.reshape(-1).detach().cpu().tolist())
        logged_rescore.extend(float(item) for item in logged_score.detach().cpu().tolist())
        target_delta.extend(float(item) for item in batch["delta_phi"].detach().cpu().tolist())

    logged_rescore_arr = np.asarray(logged_rescore, dtype=np.float64)
    random_rescore_arr = np.asarray(selector_stats["random_sample"]["rescored_phi"], dtype=np.float64)
    candidate_generated_arr = np.asarray(candidate_generated, dtype=np.float64)
    candidate_rescore_arr = np.asarray(candidate_rescore, dtype=np.float64)
    candidate_gap_arr = np.asarray(candidate_gap, dtype=np.float64)
    candidate_smooth_arr = np.asarray(candidate_smooth, dtype=np.float64)
    metrics = {
        "num_examples": float(len(rows)),
        "num_samples": float(num_samples),
        "denoise_steps": float(sample_steps),
        "rescore_batch_size": float(rescore_batch_size),
        "generated_weight": float(generated_weight),
        "rescore_weight": float(rescore_weight),
        "gap_weight": float(gap_weight),
        "smoothness_weight": float(smoothness_weight),
        "logged_rescored_phi_mean": _mean(logged_rescore),
        "target_delta_phi_mean": _mean(target_delta),
        "candidate_generated_phi_mean": _mean(candidate_generated),
        "candidate_rescored_phi_mean": _mean(candidate_rescore),
        "candidate_generation_rescore_abs_gap_mean": _mean(candidate_gap),
        "candidate_action_smoothness_mean": _mean(candidate_smooth),
        "candidate_generated_rescore_corr": _pearson_corr(candidate_generated_arr, candidate_rescore_arr),
    }
    if candidate_generated_arr.size:
        metrics["candidate_generated_phi_std"] = float(np.std(candidate_generated_arr))
        metrics["candidate_rescored_phi_std"] = float(np.std(candidate_rescore_arr))
        metrics["candidate_generation_rescore_abs_gap_std"] = float(np.std(candidate_gap_arr))
        metrics["candidate_action_smoothness_std"] = float(np.std(candidate_smooth_arr))
    for name in selector_names:
        stats = selector_stats[name]
        generated_arr = np.asarray(stats["generated_phi"], dtype=np.float64)
        rescore_arr = np.asarray(stats["rescored_phi"], dtype=np.float64)
        metrics[f"{name}_selection_score_mean"] = _mean(stats["selection_score"])
        metrics[f"{name}_generated_phi_mean"] = _mean(stats["generated_phi"])
        metrics[f"{name}_rescored_phi_mean"] = _mean(stats["rescored_phi"])
        metrics[f"{name}_minus_random_rescored_phi"] = _mean((rescore_arr - random_rescore_arr).tolist())
        metrics[f"{name}_minus_logged_rescored_phi"] = _mean((rescore_arr - logged_rescore_arr).tolist())
        metrics[f"{name}_beats_random_rescore_rate"] = (
            float(np.mean(rescore_arr > random_rescore_arr)) if rescore_arr.size else 0.0
        )
        metrics[f"{name}_beats_logged_rescore_rate"] = (
            float(np.mean(rescore_arr > logged_rescore_arr)) if rescore_arr.size else 0.0
        )
        metrics[f"{name}_action_mse_to_logged_mean"] = _mean(stats["action_mse_to_logged"])
        metrics[f"{name}_action_smoothness_mean"] = _mean(stats["action_smoothness"])
        metrics[f"{name}_generation_rescore_abs_gap_mean"] = _mean(stats["generation_rescore_gap"])
        if name != "random_sample":
            random_generated_arr = np.asarray(selector_stats["random_sample"]["generated_phi"], dtype=np.float64)
            metrics[f"{name}_minus_random_generated_phi"] = _mean((generated_arr - random_generated_arr).tolist())

    generated_alias = selector_stats["max_generated_phi"]
    generated_alias_rescore_arr = np.asarray(generated_alias["rescored_phi"], dtype=np.float64)
    generated_alias_generated_arr = np.asarray(generated_alias["generated_phi"], dtype=np.float64)
    random_generated_arr = np.asarray(selector_stats["random_sample"]["generated_phi"], dtype=np.float64)
    metrics.update(
        {
            "selected_generated_phi_mean": _mean(generated_alias["generated_phi"]),
            "random_generated_phi_mean": _mean(selector_stats["random_sample"]["generated_phi"]),
            "selected_minus_random_generated_phi": _mean(
                (generated_alias_generated_arr - random_generated_arr).tolist()
            ),
            "selected_rescored_phi_mean": _mean(generated_alias["rescored_phi"]),
            "random_rescored_phi_mean": _mean(selector_stats["random_sample"]["rescored_phi"]),
            "selected_minus_random_rescored_phi": _mean(
                (generated_alias_rescore_arr - random_rescore_arr).tolist()
            ),
            "selected_minus_logged_rescored_phi": _mean(
                (generated_alias_rescore_arr - logged_rescore_arr).tolist()
            ),
            "selected_beats_random_rescore_rate": (
                float(np.mean(generated_alias_rescore_arr > random_rescore_arr))
                if generated_alias_rescore_arr.size
                else 0.0
            ),
            "selected_beats_logged_rescore_rate": (
                float(np.mean(generated_alias_rescore_arr > logged_rescore_arr))
                if generated_alias_rescore_arr.size
                else 0.0
            ),
            "selected_action_mse_to_logged_mean": _mean(generated_alias["action_mse_to_logged"]),
            "random_action_mse_to_logged_mean": _mean(selector_stats["random_sample"]["action_mse_to_logged"]),
            "selected_action_smoothness_mean": _mean(generated_alias["action_smoothness"]),
            "random_action_smoothness_mean": _mean(selector_stats["random_sample"]["action_smoothness"]),
            "generation_rescore_abs_gap_mean": _mean(generated_alias["generation_rescore_gap"]),
        }
    )

    with (output_path / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    if rows:
        with (output_path / "selection_rows.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    with (output_path / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint": str(config.get("_checkpoint", "")),
                "num_samples": int(num_samples),
                "denoise_steps": int(sample_steps),
                "max_batches": int(max_batches),
                "split": str(split),
                "rescore_batch_size": int(rescore_batch_size),
                "selector_names": selector_names,
                "score_weights": {
                    "generated": float(generated_weight),
                    "rescore": float(rescore_weight),
                    "gap": float(gap_weight),
                    "smoothness": float(smoothness_weight),
                },
            },
            handle,
            indent=2,
            sort_keys=True,
        )
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample imagined futures from a joint-flow PP-WAM checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--denoise-steps", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_joint_flow_checkpoint(args.checkpoint, device)
    config["_checkpoint"] = str(args.checkpoint)
    loaders = make_joint_flow_loaders(config)
    run_dir = Path(args.checkpoint).resolve().parent
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else run_dir / f"sample_select_{args.split}_n{int(args.num_samples)}"
    )
    metrics = evaluate_sample_selection(
        model,
        loaders[args.split],
        config,
        device,
        output_dir=output_dir,
        num_samples=int(args.num_samples),
        denoise_steps=(int(args.denoise_steps) if int(args.denoise_steps) > 0 else None),
        max_batches=int(args.max_batches),
        seed=int(args.seed),
        split=str(args.split),
        rescore_batch_size=int(args.rescore_batch_size),
        generated_weight=float(args.generated_weight),
        rescore_weight=float(args.rescore_weight),
        gap_weight=float(args.gap_weight),
        smoothness_weight=float(args.smoothness_weight),
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
