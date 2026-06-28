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
