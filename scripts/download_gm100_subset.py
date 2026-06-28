#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_REPO_ID = "rhos-ai/gm100-cobotmagic-lerobot"
DEFAULT_OUTPUT_DIR = "/data/WAM/raw/gm100-cobotmagic-lerobot_subset"


@dataclass(frozen=True)
class SelectedEpisode:
    task_id: str
    episode_id: str


@dataclass(frozen=True)
class DownloadPlan:
    repo_id: str
    revision: str
    output_dir: str
    selected_tasks: list[str]
    selected_episodes: list[SelectedEpisode]
    allow_patterns: list[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download a small raw GM-100 LeRobot subset from Hugging Face.",
    )
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--tasks", type=int, default=20, help="Number of task folders to select.")
    parser.add_argument("--episodes-per-task", type=int, default=5)
    parser.add_argument("--task-offset", type=int, default=0, help="Offset after sorting task folders.")
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Comma-separated task folder names. Overrides --tasks and --task-offset.",
    )
    parser.add_argument("--random-tasks", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--token", default=None, help="Optional Hugging Face token.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def require_huggingface_hub():
    try:
        from huggingface_hub import HfApi
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install with "
            "`python -m pip install huggingface_hub`."
        ) from exc
    return HfApi, snapshot_download


def natural_task_key(path: str) -> tuple[int, str]:
    match = re.search(r"task_(\d+)$", path)
    if not match:
        return (10**9, path)
    return (int(match.group(1)), path)


def natural_episode_key(path: str) -> tuple[int, str]:
    match = re.search(r"episode_(\d+)\.parquet$", path)
    if not match:
        return (10**9, path)
    return (int(match.group(1)), path)


def episode_id_from_path(path: str) -> str:
    match = re.search(r"(episode_\d+)\.parquet$", path)
    if not match:
        raise ValueError(f"Cannot parse episode id from {path}")
    return match.group(1)


def item_path(item: object) -> str:
    path = getattr(item, "path", None)
    if not isinstance(path, str):
        raise TypeError(f"Unexpected Hugging Face repo-tree item without path: {item!r}")
    return path


def select_tasks(
    available_tasks: Iterable[str],
    task_ids: str | None,
    count: int,
    offset: int,
    random_tasks: bool,
    seed: int,
) -> list[str]:
    tasks = sorted(set(available_tasks), key=natural_task_key)
    if task_ids:
        requested = [task.strip() for task in task_ids.split(",") if task.strip()]
        missing = sorted(set(requested) - set(tasks), key=natural_task_key)
        if missing:
            raise ValueError(f"Requested task ids not found: {missing}")
        return requested
    if random_tasks:
        rng = random.Random(seed)
        shuffled = tasks[:]
        rng.shuffle(shuffled)
        return sorted(shuffled[:count], key=natural_task_key)
    return tasks[offset : offset + count]


def make_allow_patterns(selected: list[SelectedEpisode]) -> list[str]:
    patterns: list[str] = []
    selected_tasks = sorted({item.task_id for item in selected}, key=natural_task_key)
    for task in selected_tasks:
        patterns.extend(
            [
                f"{task}/meta/**",
            ]
        )
    for item in selected:
        patterns.extend(
            [
                f"{item.task_id}/data/chunk-000/{item.episode_id}.parquet",
                f"{item.task_id}/videos/chunk-000/*/{item.episode_id}.mp4",
            ]
        )
    return patterns


def build_plan(
    api,
    repo_id: str,
    revision: str,
    output_dir: str,
    tasks: int,
    episodes_per_task: int,
    task_offset: int,
    task_ids: str | None,
    random_tasks: bool,
    seed: int,
    token: str | None,
) -> DownloadPlan:
    root_items = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        path_in_repo="",
        recursive=False,
        token=token,
    )
    available_tasks = [item_path(item) for item in root_items if item_path(item).startswith("task_")]
    selected_tasks = select_tasks(
        available_tasks=available_tasks,
        task_ids=task_ids,
        count=tasks,
        offset=task_offset,
        random_tasks=random_tasks,
        seed=seed,
    )
    if not selected_tasks:
        raise ValueError("No task folders selected.")

    selected_episodes: list[SelectedEpisode] = []
    for task in selected_tasks:
        data_items = api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            path_in_repo=f"{task}/data/chunk-000",
            recursive=False,
            token=token,
        )
        parquet_paths = sorted(
            [item_path(item) for item in data_items if item_path(item).endswith(".parquet")],
            key=natural_episode_key,
        )
        if len(parquet_paths) < episodes_per_task:
            raise ValueError(
                f"{task} only has {len(parquet_paths)} parquet episodes; "
                f"requested {episodes_per_task}."
            )
        selected_episodes.extend(
            SelectedEpisode(task_id=task, episode_id=episode_id_from_path(path))
            for path in parquet_paths[:episodes_per_task]
        )

    return DownloadPlan(
        repo_id=repo_id,
        revision=revision,
        output_dir=output_dir,
        selected_tasks=selected_tasks,
        selected_episodes=selected_episodes,
        allow_patterns=make_allow_patterns(selected_episodes),
    )


def write_manifest(plan: DownloadPlan, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(plan)
    payload["selected_episodes"] = [asdict(item) for item in plan.selected_episodes]
    with (output_dir / "gm100_subset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def main() -> None:
    args = build_parser().parse_args()
    HfApi, snapshot_download = require_huggingface_hub()
    api = HfApi()
    plan = build_plan(
        api=api,
        repo_id=args.repo_id,
        revision=args.revision,
        output_dir=args.output_dir,
        tasks=args.tasks,
        episodes_per_task=args.episodes_per_task,
        task_offset=args.task_offset,
        task_ids=args.task_ids,
        random_tasks=args.random_tasks,
        seed=args.seed,
        token=args.token,
    )

    print(
        f"selected {len(plan.selected_tasks)} tasks and "
        f"{len(plan.selected_episodes)} episodes from {plan.repo_id}@{plan.revision}"
    )
    print(f"output_dir={plan.output_dir}")
    if args.dry_run:
        print(json.dumps(asdict(plan), indent=2, sort_keys=True))
        return

    snapshot_download(
        repo_id=plan.repo_id,
        repo_type="dataset",
        revision=plan.revision,
        local_dir=plan.output_dir,
        cache_dir=args.cache_dir,
        allow_patterns=plan.allow_patterns,
        max_workers=args.max_workers,
        token=args.token,
    )
    write_manifest(plan, plan.output_dir)
    print(f"wrote manifest to {Path(plan.output_dir) / 'gm100_subset_manifest.json'}")


if __name__ == "__main__":
    main()
