import subprocess
import sys

from mvp0.create_toy_episodes import create_toy_episodes


def test_create_toy_episodes_writes_expected_files(tmp_path):
    episode_ids = create_toy_episodes(tmp_path / "episodes", num_episodes=2, num_frames=16)

    assert episode_ids == ["toy_ep0000", "toy_ep0001"]
    assert (tmp_path / "episodes" / "toy_ep0000" / "meta.json").exists()
    assert (tmp_path / "episodes" / "toy_ep0000" / "arrays.npz").exists()
    assert (tmp_path / "episodes" / "toy_ep0000" / "labels.json").exists()


def test_smoke_cli_runs_end_to_end(tmp_path):
    root = tmp_path / "smoke"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "mvp0.smoke",
            "--root",
            str(root),
            "--num-episodes",
            "5",
            "--num-frames",
            "16",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "smoke complete" in result.stdout
    assert (root / "windows" / "windows.jsonl").exists()
    assert (root / "features" / "toy_ep0000.npz").exists()
    assert (root / "counterfactuals" / "simple_pairs.npz").exists()
    assert (root / "outputs" / "obs_action_stage_cf" / "best.pt").exists()
    assert (root / "outputs" / "obs_action_stage_cf" / "eval" / "stage_sensitivity.csv").exists()
    assert (root / "outputs" / "report" / "summary.csv").exists()

