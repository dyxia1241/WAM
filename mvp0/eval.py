from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from mvp0.counterfactual import make_negative_batch
from mvp0.manifest import write_manifest
from mvp0.metrics import compute_metrics, summarize_by_type
from mvp0.train import PROMPT_EXPERIMENTS, batch_to_device, build_model, evaluate_model, forward_model, make_loaders


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MVP-0 toy evaluation entrypoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--negative-types",
        default="zero,reverse,shuffle,wrong_arm,scaled_0.25,scaled_1.75",
        help="Comma-separated negative types for action sensitivity.",
    )
    return parser


@torch.no_grad()
def write_predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    experiment: str,
    device: torch.device,
    path: Path,
) -> dict[str, float]:
    model.eval()
    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        offset = 0
        for batch in loader:
            batch = batch_to_device(batch, device)
            logits = forward_model(model, batch, experiment)
            pred = torch.sigmoid(logits).cpu().reshape(-1)
            target = batch["delta_phi"].cpu().reshape(-1)
            preds.append(pred)
            targets.append(target)
            for i, (pred_value, target_value) in enumerate(zip(pred.tolist(), target.tolist(), strict=True)):
                handle.write(
                    json.dumps(
                        {
                            "index": offset + i,
                            "pred_delta_phi": pred_value,
                            "target_delta_phi": target_value,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
            offset += len(pred)
    return compute_metrics(torch.cat(preds), torch.cat(targets))


@torch.no_grad()
def write_action_sensitivity(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    experiment: str,
    device: torch.device,
    negative_types: list[str],
    path: Path,
) -> dict[str, float]:
    if experiment == "time_prior":
        return {}

    model.eval()
    rows: list[dict[str, float | int | str]] = []
    all_pos: list[torch.Tensor] = []
    all_neg: list[torch.Tensor] = []
    all_types: list[str] = []
    offset = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        for negative_type in negative_types:
            try:
                paired = make_negative_batch(batch, kind=negative_type)
            except ValueError:
                continue
            pos = torch.sigmoid(forward_model(model, paired.positive, experiment)).cpu().reshape(-1)
            neg = torch.sigmoid(forward_model(model, paired.negative, experiment)).cpu().reshape(-1)
            margin = pos - neg
            all_pos.append(pos)
            all_neg.append(neg)
            all_types.extend([negative_type] * len(pos))
            for i in range(len(pos)):
                rows.append(
                    {
                        "index": offset + i,
                        "negative_type": negative_type,
                        "pos_delta_phi": float(pos[i]),
                        "neg_delta_phi": float(neg[i]),
                        "margin": float(margin[i]),
                        "is_correct": int(pos[i] > neg[i]),
                    }
                )
        offset += int(batch["delta_phi"].shape[0])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["index", "negative_type", "pos_delta_phi", "neg_delta_phi", "margin", "is_correct"],
        )
        writer.writeheader()
        writer.writerows(rows)

    if not all_pos:
        return {}
    return summarize_by_type(torch.cat(all_pos), torch.cat(all_neg), all_types)


@torch.no_grad()
def write_stage_sensitivity(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    experiment: str,
    device: torch.device,
    path: Path,
    num_stages: int = 5,
) -> dict[str, float]:
    if experiment == "time_prior" or experiment in PROMPT_EXPERIMENTS:
        return {}

    model.eval()
    rows: list[dict[str, float | int | str]] = []
    margins_by_type: dict[str, list[float]] = {"previous": [], "next": [], "random": []}
    win_by_type: dict[str, list[float]] = {"previous": [], "next": [], "random": []}
    high_wrong: list[float] = []
    offset = 0
    for batch in loader:
        batch = batch_to_device(batch, device)
        true_score = torch.sigmoid(forward_model(model, batch, experiment)).cpu().reshape(-1)
        stage_variants = {
            "previous": (batch["stage_id"] - 1) % num_stages,
            "next": (batch["stage_id"] + 1) % num_stages,
            "random": (batch["stage_id"] + 2) % num_stages,
        }
        for replacement_type, replacement_stage in stage_variants.items():
            replaced = {key: value.clone() for key, value in batch.items()}
            replaced["stage_id"] = replacement_stage
            wrong_score = torch.sigmoid(forward_model(model, replaced, experiment)).cpu().reshape(-1)
            margin = true_score - wrong_score
            margins_by_type[replacement_type].extend(margin.tolist())
            win_by_type[replacement_type].extend((margin > 0).float().tolist())
            high_wrong.extend((wrong_score > true_score).float().tolist())
            for i in range(len(true_score)):
                rows.append(
                    {
                        "index": offset + i,
                        "replacement_type": replacement_type,
                        "true_stage": int(batch["stage_id"][i].detach().cpu()),
                        "replacement_stage": int(replacement_stage[i].detach().cpu()),
                        "true_delta_phi": float(true_score[i]),
                        "wrong_delta_phi": float(wrong_score[i]),
                        "margin": float(margin[i]),
                        "is_true_higher": int(margin[i] > 0),
                    }
                )
        offset += int(batch["delta_phi"].shape[0])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "index",
                "replacement_type",
                "true_stage",
                "replacement_stage",
                "true_delta_phi",
                "wrong_delta_phi",
                "margin",
                "is_true_higher",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    metrics: dict[str, float] = {}
    all_margins: list[float] = []
    for replacement_type, margins in margins_by_type.items():
        if not margins:
            continue
        all_margins.extend(margins)
        metrics[f"stage_{replacement_type}_mean_margin"] = float(sum(margins) / len(margins))
        metrics[f"stage_{replacement_type}_true_win_rate"] = float(
            sum(win_by_type[replacement_type]) / len(win_by_type[replacement_type])
        )
    if all_margins:
        metrics["true_vs_wrong_stage_margin"] = float(sum(all_margins) / len(all_margins))
        metrics["wrong_stage_high_progress_rate"] = float(sum(high_wrong) / len(high_wrong))
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    run_eval(
        checkpoint=args.checkpoint,
        split=args.split,
        output=args.output,
        negative_types=args.negative_types,
    )


def run_eval(
    checkpoint: str | Path,
    split: str = "test",
    output: str | Path | None = None,
    negative_types: str = "zero,reverse,shuffle,wrong_arm,scaled_0.25,scaled_1.75",
) -> dict[str, float]:
    checkpoint_path = Path(checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    experiment = checkpoint["experiment"]
    device = torch.device(config.get("device", "cpu"))

    loaders = make_loaders(config)
    if split not in loaders:
        raise ValueError(f"Unknown split: {split}")

    model = build_model(config, experiment).to(device)
    model.load_state_dict(checkpoint["model_state"])

    output_dir = Path(output) if output else checkpoint_path.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_model(model, loaders[split], experiment, device)
    pred_metrics = write_predictions(
        model,
        loaders[split],
        experiment,
        device,
        output_dir / "predictions.jsonl",
    )
    parsed_negative_types = [item.strip() for item in negative_types.split(",") if item.strip()]
    sensitivity_metrics = write_action_sensitivity(
        model,
        loaders[split],
        experiment,
        device,
        negative_types=parsed_negative_types,
        path=output_dir / "action_sensitivity.csv",
    )
    stage_metrics = write_stage_sensitivity(
        model,
        loaders[split],
        experiment,
        device,
        path=output_dir / "stage_sensitivity.csv",
        num_stages=int(config.get("data", {}).get("num_stages", 5)),
    )
    metrics.update({f"prediction_{key}": value for key, value in pred_metrics.items()})
    metrics.update(sensitivity_metrics)
    metrics.update(stage_metrics)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    write_manifest(
        output_dir / "manifest.json",
        kind="eval",
        config=config,
        metrics=metrics,
        experiment=experiment,
        checkpoint=str(checkpoint_path),
        split=split,
        repo_root=Path(__file__).resolve().parents[1],
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


if __name__ == "__main__":
    main()
