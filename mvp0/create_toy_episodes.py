from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def create_toy_episodes(
    output: str | Path,
    num_episodes: int = 5,
    num_frames: int = 24,
    action_dim: int = 4,
    proprio_dim: int = 4,
    num_tasks: int = 2,
    seed: int = 42,
) -> list[str]:
    if num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")
    if num_frames < 12:
        raise ValueError("num_frames must be at least 12.")
    if action_dim <= 0 or proprio_dim <= 0:
        raise ValueError("action_dim and proprio_dim must be positive.")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    episode_ids: list[str] = []
    mid = num_frames // 2
    for episode_idx in range(num_episodes):
        episode_id = f"toy_ep{episode_idx:04d}"
        task_id = f"toy_task{episode_idx % num_tasks}"
        episode_dir = output / episode_id
        episode_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed + episode_idx)

        meta = {
            "episode_id": episode_id,
            "task_id": task_id,
            "language": f"toy command for {task_id}",
            "fps": 10,
            "num_frames": num_frames,
            "cameras": ["cam0"],
            "action_dim": action_dim,
            "proprio_dim": proprio_dim,
            "action_space": "absolute",
        }
        labels = {
            "primitive_boundaries": [
                {"stage": "approach", "start": 0, "end": mid - 1},
                {"stage": "grasp", "start": mid, "end": num_frames - 1},
            ],
            "success": True,
        }
        (episode_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        (episode_dir / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True), encoding="utf-8")

        action_base = np.linspace(0.0, 1.0, num_frames, dtype=np.float32)[:, None]
        action = np.clip(
            action_base + 0.05 * rng.normal(size=(num_frames, action_dim)).astype(np.float32),
            -1.0,
            1.0,
        )
        proprio = rng.normal(size=(num_frames, proprio_dim)).astype(np.float32)
        np.savez_compressed(episode_dir / "arrays.npz", proprio=proprio, action=action)
        episode_ids.append(episode_id)
    return episode_ids


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create file-based toy episodes for WSL smoke tests.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--action-dim", type=int, default=4)
    parser.add_argument("--proprio-dim", type=int, default=4)
    parser.add_argument("--num-tasks", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    episode_ids = create_toy_episodes(
        output=args.output,
        num_episodes=args.num_episodes,
        num_frames=args.num_frames,
        action_dim=args.action_dim,
        proprio_dim=args.proprio_dim,
        num_tasks=args.num_tasks,
        seed=args.seed,
    )
    print(f"wrote {len(episode_ids)} toy episodes to {args.output}")


if __name__ == "__main__":
    main()

