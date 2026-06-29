from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download_gm100_subset.py"
SPEC = importlib.util.spec_from_file_location("download_gm100_subset", SCRIPT_PATH)
download_gm100_subset = importlib.util.module_from_spec(SPEC)
assert SPEC is not None
assert SPEC.loader is not None
sys.modules["download_gm100_subset"] = download_gm100_subset
SPEC.loader.exec_module(download_gm100_subset)


class FakeRepoItem:
    def __init__(self, path: str) -> None:
        self.path = path


class FakeHfApi:
    def list_repo_tree(
        self,
        repo_id: str,
        repo_type: str,
        revision: str,
        path_in_repo: str,
        recursive: bool,
        token: str | None = None,
    ):
        assert repo_id == "fake/gm100"
        assert repo_type == "dataset"
        assert revision == "main"
        assert recursive is False
        if path_in_repo == "":
            return [
                FakeRepoItem("README.md"),
                FakeRepoItem("task_00002"),
                FakeRepoItem("task_00001"),
                FakeRepoItem("task_00010"),
            ]
        if path_in_repo.endswith("/data/chunk-000"):
            task = path_in_repo.split("/")[0]
            return [
                FakeRepoItem(f"{task}/data/chunk-000/episode_000002.parquet"),
                FakeRepoItem(f"{task}/data/chunk-000/episode_000000.parquet"),
                FakeRepoItem(f"{task}/data/chunk-000/episode_000001.parquet"),
            ]
        if path_in_repo.endswith("/meta"):
            task = path_in_repo.split("/")[0]
            return [
                FakeRepoItem(f"{task}/meta/info.json"),
                FakeRepoItem(f"{task}/meta/tasks.jsonl"),
            ]
        if path_in_repo.endswith("/videos/chunk-000"):
            task = path_in_repo.split("/")[0]
            return [
                FakeRepoItem(f"{task}/videos/chunk-000/observation.images.camera_top"),
                FakeRepoItem(f"{task}/videos/chunk-000/observation.images.camera_wrist_left"),
                FakeRepoItem(f"{task}/videos/chunk-000/observation.images.camera_wrist_right"),
            ]
        raise AssertionError(path_in_repo)


def test_build_plan_selects_natural_task_and_episode_order() -> None:
    plan = download_gm100_subset.build_plan(
        api=FakeHfApi(),
        repo_id="fake/gm100",
        revision="main",
        output_dir="/tmp/gm100",
        tasks=2,
        episodes_per_task=2,
        task_offset=0,
        task_ids=None,
        random_tasks=False,
        random_episodes=False,
        seed=42,
        token=None,
    )

    assert plan.selected_tasks == ["task_00001", "task_00002"]
    assert [(item.task_id, item.episode_id) for item in plan.selected_episodes] == [
        ("task_00001", "episode_000000"),
        ("task_00001", "episode_000001"),
        ("task_00002", "episode_000000"),
        ("task_00002", "episode_000001"),
    ]
    assert "task_00001/meta/info.json" in plan.exact_files
    assert "task_00001/data/chunk-000/episode_000000.parquet" in plan.exact_files
    assert (
        "task_00001/videos/chunk-000/observation.images.camera_top/episode_000000.mp4"
        in plan.exact_files
    )


def test_make_allow_patterns_include_meta_data_and_all_camera_videos() -> None:
    selected = [
        download_gm100_subset.SelectedEpisode("task_00001", "episode_000000"),
        download_gm100_subset.SelectedEpisode("task_00001", "episode_000001"),
    ]

    assert download_gm100_subset.make_allow_patterns(selected) == [
        "task_00001/meta/**",
        "task_00001/data/chunk-000/episode_000000.parquet",
        "task_00001/videos/chunk-000/*/episode_000000.mp4",
        "task_00001/data/chunk-000/episode_000001.parquet",
        "task_00001/videos/chunk-000/*/episode_000001.mp4",
    ]


def test_task_id_shorthand_is_normalized() -> None:
    assert download_gm100_subset.normalize_task_id("task001") == "task_00001"
    assert download_gm100_subset.normalize_task_id("task_002") == "task_00002"
    assert download_gm100_subset.normalize_task_id("task_00010") == "task_00010"


def test_random_episode_selection_is_seeded_and_sorted() -> None:
    parquet_paths = [
        f"task_00001/data/chunk-000/episode_{episode:06d}.parquet"
        for episode in range(10)
    ]

    first = download_gm100_subset.select_episode_paths(
        parquet_paths=parquet_paths,
        episodes_per_task=2,
        random_episodes=True,
        seed=7,
        task_id="task_00001",
    )
    second = download_gm100_subset.select_episode_paths(
        parquet_paths=parquet_paths,
        episodes_per_task=2,
        random_episodes=True,
        seed=7,
        task_id="task_00001",
    )

    assert first == second
    assert first == sorted(first, key=download_gm100_subset.natural_episode_key)
    assert first != parquet_paths[:2]
