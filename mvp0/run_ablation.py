from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mvp0.config import apply_overrides, load_config
from mvp0.train import EXPERIMENTS, train


DEFAULT_ABLATIONS = (
    "time_prior",
    "obs_stage",
    "obs_action",
    "obs_action_stage",
    "obs_action_stage_cf",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MVP-0 toy ablations sequentially.")
    parser.add_argument("--config", default="mvp0/configs/debug.yaml")
    parser.add_argument("--experiments", default=",".join(DEFAULT_ABLATIONS))
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("overrides", nargs="*", help="Optional key=value overrides shared by all experiments.")
    return parser


def run_ablation(config: dict[str, Any], experiments: list[str], output_dir: str | Path) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    for experiment in experiments:
        if experiment not in EXPERIMENTS:
            raise ValueError(f"Unknown experiment: {experiment}")
        run_config = dict(config)
        run_config["experiment"] = experiment
        run_config["output_dir"] = str(output_dir)
        results[experiment] = train(run_config)
    return results


def main() -> None:
    args = build_parser().parse_args()
    config = apply_overrides(load_config(args.config), args.overrides)
    experiments = [item.strip() for item in args.experiments.split(",") if item.strip()]
    results = run_ablation(config, experiments=experiments, output_dir=args.output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ablation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

