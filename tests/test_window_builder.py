from ppwam.labels import PrimitiveBoundary
from ppwam.schemas import EpisodeMeta, EpisodeSpec
from ppwam.window_builder import build_windows, episode_to_split, split_episodes


def _episode(episode_id: str, task_id: str = "task") -> EpisodeSpec:
    return EpisodeSpec(
        meta=EpisodeMeta(
            episode_id=episode_id,
            task_id=task_id,
            num_frames=20,
            action_dim=4,
            proprio_dim=4,
        ),
        boundaries=(
            PrimitiveBoundary(stage="approach", start=0, end=9),
            PrimitiveBoundary(stage="grasp", start=10, end=19),
        ),
    )


def test_split_episodes_has_no_leakage():
    split = split_episodes([f"ep{i}" for i in range(10)], seed=0)
    all_ids = split["train"] + split["val"] + split["test"]

    assert sorted(all_ids) == [f"ep{i}" for i in range(10)]
    assert len(all_ids) == len(set(all_ids))


def test_build_windows_indices_do_not_exceed_episode():
    episodes = [_episode("ep0")]
    split_map = {"ep0": "train"}

    windows = build_windows(episodes, split_map, history=4, horizon=4, stride=2)

    assert windows
    for window in windows:
        assert min(window.history_indices) >= 0
        assert max(window.future_indices) < 20
        assert window.split == "train"


def test_cross_boundary_can_be_excluded():
    episodes = [_episode("ep0")]
    split_map = {"ep0": "train"}

    kept = build_windows(episodes, split_map, history=4, horizon=4, stride=1, exclude_cross_boundary=True)

    assert kept
    assert all(not window.cross_boundary for window in kept)


def test_episode_to_split_rejects_duplicates():
    try:
        episode_to_split({"train": ["ep0"], "test": ["ep0"]})
    except ValueError as exc:
        assert "multiple splits" in str(exc)
    else:
        raise AssertionError("Expected duplicate episode split to fail.")

