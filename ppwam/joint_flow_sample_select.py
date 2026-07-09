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

    scores = _phi_score(phi_state, reduce=phi_reduce, clamp=True)
    return {
        "action": action_state.reshape(batch_size, int(num_samples), horizon, action_dim),
        "future_obs": future_obs_state.reshape(batch_size, int(num_samples), *future_obs_state.shape[1:]),
        "phi_state": phi_state.reshape(batch_size, int(num_samples), phi_state.shape[-1]),
        "generated_phi": scores.reshape(batch_size, int(num_samples)),
    }


def _gather_candidates(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch = torch.arange(values.shape[0], device=values.device)
    return values[batch, indices]


def _action_smoothness(action: torch.Tensor) -> torch.Tensor:
    if action.shape[1] <= 1:
        return torch.zeros((action.shape[0],), dtype=action.dtype, device=action.device)
    return torch.mean((action[:, 1:] - action[:, :-1]) ** 2, dim=(1, 2))


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
) -> dict[str, float]:
    runtime = joint_flow_runtime_options(config)
    sample_steps = int(denoise_steps if denoise_steps is not None else runtime["score_denoise_steps"])
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed) + 10_003)
    model.eval()

    rows: list[dict[str, float | int]] = []
    selected_generated: list[float] = []
    random_generated: list[float] = []
    selected_rescore: list[float] = []
    random_rescore: list[float] = []
    logged_rescore: list[float] = []
    target_delta: list[float] = []
    selected_mse: list[float] = []
    random_mse: list[float] = []
    selected_smooth: list[float] = []
    random_smooth: list[float] = []
    consistency_gap: list[float] = []

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
            generator=generator,
        )
        generated_phi = futures["generated_phi"]
        best_indices = torch.argmax(generated_phi, dim=1)
        random_indices = torch.zeros_like(best_indices)
        selected_action = _gather_candidates(futures["action"], best_indices)
        random_action = _gather_candidates(futures["action"], random_indices)

        selected_score = score_action(
            model,
            batch,
            action_chunk=selected_action,
            denoise_steps=runtime["score_denoise_steps"],
            phi_tokens=runtime["phi_tokens"],
            phi_reduce=runtime["phi_reduce"],
            future_obs_init=runtime["future_obs_init"],
        )
        random_score = score_action(
            model,
            batch,
            action_chunk=random_action,
            denoise_steps=runtime["score_denoise_steps"],
            phi_tokens=runtime["phi_tokens"],
            phi_reduce=runtime["phi_reduce"],
            future_obs_init=runtime["future_obs_init"],
        )
        logged_score = score_action(
            model,
            batch,
            action_chunk=batch["action_chunk"],
            denoise_steps=runtime["score_denoise_steps"],
            phi_tokens=runtime["phi_tokens"],
            phi_reduce=runtime["phi_reduce"],
            future_obs_init=runtime["future_obs_init"],
        )

        selected_gen = generated_phi.gather(1, best_indices.reshape(-1, 1)).squeeze(1)
        random_gen = generated_phi[:, 0]
        selected_mse_batch = torch.mean((selected_action - batch["action_chunk"]) ** 2, dim=(1, 2))
        random_mse_batch = torch.mean((random_action - batch["action_chunk"]) ** 2, dim=(1, 2))
        selected_smooth_batch = _action_smoothness(selected_action)
        random_smooth_batch = _action_smoothness(random_action)
        gap_batch = torch.abs(selected_gen - selected_score)

        for local_index in range(int(batch["action_chunk"].shape[0])):
            row = {
                "index": global_index,
                "selected_sample": int(best_indices[local_index].detach().cpu()),
                "target_delta_phi": float(batch["delta_phi"][local_index].detach().cpu()),
                "selected_generated_phi": float(selected_gen[local_index].detach().cpu()),
                "random_generated_phi": float(random_gen[local_index].detach().cpu()),
                "selected_rescored_phi": float(selected_score[local_index].detach().cpu()),
                "random_rescored_phi": float(random_score[local_index].detach().cpu()),
                "logged_rescored_phi": float(logged_score[local_index].detach().cpu()),
                "selected_action_mse_to_logged": float(selected_mse_batch[local_index].detach().cpu()),
                "random_action_mse_to_logged": float(random_mse_batch[local_index].detach().cpu()),
                "selected_action_smoothness": float(selected_smooth_batch[local_index].detach().cpu()),
                "random_action_smoothness": float(random_smooth_batch[local_index].detach().cpu()),
                "selected_generation_rescore_gap": float(gap_batch[local_index].detach().cpu()),
            }
            rows.append(row)
            global_index += 1

        selected_generated.extend(float(item) for item in selected_gen.detach().cpu().tolist())
        random_generated.extend(float(item) for item in random_gen.detach().cpu().tolist())
        selected_rescore.extend(float(item) for item in selected_score.detach().cpu().tolist())
        random_rescore.extend(float(item) for item in random_score.detach().cpu().tolist())
        logged_rescore.extend(float(item) for item in logged_score.detach().cpu().tolist())
        target_delta.extend(float(item) for item in batch["delta_phi"].detach().cpu().tolist())
        selected_mse.extend(float(item) for item in selected_mse_batch.detach().cpu().tolist())
        random_mse.extend(float(item) for item in random_mse_batch.detach().cpu().tolist())
        selected_smooth.extend(float(item) for item in selected_smooth_batch.detach().cpu().tolist())
        random_smooth.extend(float(item) for item in random_smooth_batch.detach().cpu().tolist())
        consistency_gap.extend(float(item) for item in gap_batch.detach().cpu().tolist())

    selected_rescore_arr = np.asarray(selected_rescore, dtype=np.float64)
    random_rescore_arr = np.asarray(random_rescore, dtype=np.float64)
    logged_rescore_arr = np.asarray(logged_rescore, dtype=np.float64)
    selected_generated_arr = np.asarray(selected_generated, dtype=np.float64)
    random_generated_arr = np.asarray(random_generated, dtype=np.float64)
    metrics = {
        "num_examples": float(len(rows)),
        "num_samples": float(num_samples),
        "denoise_steps": float(sample_steps),
        "selected_generated_phi_mean": _mean(selected_generated),
        "random_generated_phi_mean": _mean(random_generated),
        "selected_minus_random_generated_phi": _mean((selected_generated_arr - random_generated_arr).tolist()),
        "selected_rescored_phi_mean": _mean(selected_rescore),
        "random_rescored_phi_mean": _mean(random_rescore),
        "logged_rescored_phi_mean": _mean(logged_rescore),
        "target_delta_phi_mean": _mean(target_delta),
        "selected_minus_random_rescored_phi": _mean((selected_rescore_arr - random_rescore_arr).tolist()),
        "selected_minus_logged_rescored_phi": _mean((selected_rescore_arr - logged_rescore_arr).tolist()),
        "selected_beats_random_rescore_rate": float(np.mean(selected_rescore_arr > random_rescore_arr))
        if selected_rescore_arr.size
        else 0.0,
        "selected_beats_logged_rescore_rate": float(np.mean(selected_rescore_arr > logged_rescore_arr))
        if selected_rescore_arr.size
        else 0.0,
        "selected_action_mse_to_logged_mean": _mean(selected_mse),
        "random_action_mse_to_logged_mean": _mean(random_mse),
        "selected_action_smoothness_mean": _mean(selected_smooth),
        "random_action_smoothness_mean": _mean(random_smooth),
        "generation_rescore_abs_gap_mean": _mean(consistency_gap),
    }

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
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
