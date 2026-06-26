from __future__ import annotations

import argparse
from pathlib import Path

from mvp0.config import apply_overrides, load_config
from mvp0.create_toy_episodes import create_toy_episodes
from mvp0.eval import run_eval
from mvp0.extract_vision_features import extract_mock_features
from mvp0.make_counterfactuals import make_counterfactuals
from mvp0.plot import run_plot
from mvp0.prepare_windows import prepare_windows
from mvp0.reports import collect_runs, write_report
from mvp0.run_ablation import run_ablation


def run_smoke(
    root: str | Path,
    config_path: str | Path = "mvp0/configs/debug.yaml",
    num_episodes: int = 5,
    num_frames: int = 24,
) -> None:
    root = Path(root)
    episodes = root / "episodes"
    windows = root / "windows"
    features = root / "features"
    counterfactuals = root / "counterfactuals"
    outputs = root / "outputs"

    create_toy_episodes(episodes, num_episodes=num_episodes, num_frames=num_frames)
    prepare_windows(
        episodes,
        windows,
        history=4,
        horizon=4,
        stride=2,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
    )
    extract_mock_features(episodes, features, feature_dim=8, seed=42)
    make_counterfactuals(windows, counterfactuals)

    config = apply_overrides(
        load_config(config_path),
        [
            "train.max_epochs=1",
            "data.batch_size=4",
            "data.horizon=4",
            "data.action_dim=4",
            "data.proprio_dim=4",
            f"data.windows_dir={windows}",
            f"data.episodes_dir={episodes}",
            f"data.features_dir={features}",
            "features.feature_dim=8",
            "model.hidden_dim=32",
            "model.transformer_layers=1",
            "model.transformer_heads=4",
            f"output_dir={outputs}",
        ],
    )
    run_ablation(config, ["time_prior", "obs_stage", "obs_action_stage_cf"], outputs)

    checkpoint = outputs / "obs_action_stage_cf" / "best.pt"
    eval_dir = outputs / "obs_action_stage_cf" / "eval"

    run_eval(checkpoint=checkpoint, split="test", output=eval_dir)
    run_plot(eval_dir)

    write_report(collect_runs(outputs), outputs / "report")
    print(f"smoke complete: {root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an end-to-end WSL toy smoke pipeline.")
    parser.add_argument("--root", default="/tmp/wam_smoke")
    parser.add_argument("--config", default="mvp0/configs/debug.yaml")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=24)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_smoke(
        root=args.root,
        config_path=args.config,
        num_episodes=args.num_episodes,
        num_frames=args.num_frames,
    )


if __name__ == "__main__":
    main()
