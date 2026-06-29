#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import quote


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
    exact_files: list[str]


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
    parser.add_argument("--random-episodes", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--token", default=None, help="Optional Hugging Face token.")
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
        help="Dataset endpoint. Use https://hf-mirror.com when huggingface.co is unreachable.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def require_huggingface_hub():
    try:
        from huggingface_hub import HfApi
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install with "
            "`python -m pip install huggingface_hub`."
        ) from exc
    return HfApi


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
        requested = [normalize_task_id(task) for task in task_ids.split(",") if task.strip()]
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


def normalize_task_id(task_id: str) -> str:
    stripped = task_id.strip()
    match = re.fullmatch(r"task_?0*(\d+)", stripped)
    if match:
        return f"task_{int(match.group(1)):05d}"
    return stripped


def select_episode_paths(
    parquet_paths: list[str],
    episodes_per_task: int,
    random_episodes: bool,
    seed: int,
    task_id: str,
) -> list[str]:
    if len(parquet_paths) < episodes_per_task:
        raise ValueError(
            f"{task_id} only has {len(parquet_paths)} parquet episodes; "
            f"requested {episodes_per_task}."
        )
    if not random_episodes:
        return parquet_paths[:episodes_per_task]
    rng = random.Random(f"{seed}:{task_id}")
    selected = rng.sample(parquet_paths, episodes_per_task)
    return sorted(selected, key=natural_episode_key)


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


def list_paths(api, repo_id: str, revision: str, path: str, token: str | None) -> list[str]:
    items = api.list_repo_tree(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        path_in_repo=path,
        recursive=False,
        token=token,
    )
    return [item_path(item) for item in items]


def unique_sorted(paths: Iterable[str]) -> list[str]:
    return sorted(set(paths))


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
    random_episodes: bool,
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
    exact_files: list[str] = []
    for task in selected_tasks:
        exact_files.extend(
            path for path in list_paths(api, repo_id, revision, f"{task}/meta", token) if not path.endswith("/")
        )
        camera_dirs = list_paths(api, repo_id, revision, f"{task}/videos/chunk-000", token)
        data_items = list_paths(api, repo_id, revision, f"{task}/data/chunk-000", token)
        parquet_paths = sorted(
            [path for path in data_items if path.endswith(".parquet")],
            key=natural_episode_key,
        )
        selected_paths = select_episode_paths(
            parquet_paths=parquet_paths,
            episodes_per_task=episodes_per_task,
            random_episodes=random_episodes,
            seed=seed,
            task_id=task,
        )
        exact_files.extend(selected_paths)
        selected_episodes.extend(
            SelectedEpisode(task_id=task, episode_id=episode_id_from_path(path))
            for path in selected_paths
        )
        for episode_path in selected_paths:
            episode_id = episode_id_from_path(episode_path)
            exact_files.extend(
                f"{camera_dir}/{episode_id}.mp4"
                for camera_dir in camera_dirs
                if camera_dir.startswith(f"{task}/videos/chunk-000/")
            )

    return DownloadPlan(
        repo_id=repo_id,
        revision=revision,
        output_dir=output_dir,
        selected_tasks=selected_tasks,
        selected_episodes=selected_episodes,
        allow_patterns=make_allow_patterns(selected_episodes),
        exact_files=unique_sorted(exact_files),
    )


def write_manifest(plan: DownloadPlan, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = asdict(plan)
    payload["selected_episodes"] = [asdict(item) for item in plan.selected_episodes]
    with (output_dir / "gm100_subset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def file_url(endpoint: str, repo_id: str, revision: str, path: str) -> str:
    quoted_path = quote(path, safe="/")
    return f"{endpoint.rstrip('/')}/datasets/{repo_id}/resolve/{revision}/{quoted_path}"


def download_exact_file(
    endpoint: str,
    repo_id: str,
    revision: str,
    path: str,
    output_dir: str | Path,
    token: str | None,
    overwrite: bool,
) -> bool:
    destination = Path(output_dir) / path
    if destination.exists() and not overwrite:
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")
    command = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "3",
        "--connect-timeout",
        "20",
        "--max-time",
        "900",
        "--output",
        str(tmp_path),
        file_url(endpoint, repo_id, revision, path),
    ]
    if token:
        command[1:1] = ["--header", f"Authorization: Bearer {token}"]
    subprocess.run(command, check=True)
    tmp_path.replace(destination)
    return True


def download_exact_files(
    plan: DownloadPlan,
    endpoint: str,
    token: str | None,
    overwrite: bool,
) -> None:
    total = len(plan.exact_files)
    for idx, path in enumerate(plan.exact_files, start=1):
        wrote = download_exact_file(
            endpoint=endpoint,
            repo_id=plan.repo_id,
            revision=plan.revision,
            path=path,
            output_dir=plan.output_dir,
            token=token,
            overwrite=overwrite,
        )
        action = "downloaded" if wrote else "exists"
        print(f"[{idx}/{total}] {action} {path}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    HfApi = require_huggingface_hub()
    api = HfApi(endpoint=args.endpoint)
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
        random_episodes=args.random_episodes,
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

    download_exact_files(
        plan=plan,
        endpoint=args.endpoint,
        token=args.token,
        overwrite=args.overwrite,
    )
    write_manifest(plan, plan.output_dir)
    print(f"wrote manifest to {Path(plan.output_dir) / 'gm100_subset_manifest.json'}")


if __name__ == "__main__":
    main()
