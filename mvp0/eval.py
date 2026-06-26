from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from mvp0.metrics import compute_metrics
from mvp0.train import batch_to_device, build_model, evaluate_model, forward_model, make_loaders


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MVP-0 toy evaluation entrypoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default=None)
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


def main() -> None:
    args = build_parser().parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    experiment = checkpoint["experiment"]
    device = torch.device(config.get("device", "cpu"))

    loaders = make_loaders(config)
    if args.split not in loaders:
        raise ValueError(f"Unknown split: {args.split}")

    model = build_model(config, experiment).to(device)
    model.load_state_dict(checkpoint["model_state"])

    output_dir = Path(args.output) if args.output else checkpoint_path.parent / "eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = evaluate_model(model, loaders[args.split], experiment, device)
    pred_metrics = write_predictions(
        model,
        loaders[args.split],
        experiment,
        device,
        output_dir / "predictions.jsonl",
    )
    metrics.update({f"prediction_{key}": value for key, value in pred_metrics.items()})
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

