import json
import subprocess
import sys

import numpy as np

from ppwam.make_counterfactuals import generate_simple_pairs, make_counterfactuals


def _windows():
    return [
        {"window_id": "w0", "split": "train", "stage": "approach"},
        {"window_id": "w1", "split": "train", "stage": "approach"},
        {"window_id": "w2", "split": "train", "stage": "grasp"},
        {"window_id": "w3", "split": "train", "stage": "grasp"},
        {"window_id": "w4", "split": "test", "stage": "approach"},
        {"window_id": "w5", "split": "test", "stage": "approach"},
    ]


def test_generate_simple_pairs_shuffle_stays_in_split_and_stage():
    windows = _windows()

    pairs = generate_simple_pairs(windows, negative_types=("shuffle",), seed=0)

    assert len(pairs["positive_index"]) == len(windows)
    for pos_idx, repl_idx in zip(pairs["positive_index"], pairs["replacement_index"], strict=True):
        pos = windows[int(pos_idx)]
        repl = windows[int(repl_idx)]
        assert int(pos_idx) != int(repl_idx)
        assert pos["split"] == repl["split"]
        assert pos["stage"] == repl["stage"]


def test_generate_simple_pairs_records_non_retrieval_negatives():
    pairs = generate_simple_pairs(_windows(), negative_types=("zero", "reverse", "scaled_1.75"), seed=0)

    assert len(pairs["positive_index"]) == 18
    assert set(pairs["negative_kind"].tolist()) == {"zero", "reverse", "scaled_1.75"}
    assert np.all(pairs["replacement_index"] == -1)


def test_make_counterfactuals_writes_npz_and_index(tmp_path):
    windows_path = tmp_path / "windows.jsonl"
    with windows_path.open("w", encoding="utf-8") as handle:
        for window in _windows():
            handle.write(json.dumps(window) + "\n")
    output_dir = tmp_path / "counterfactuals"

    pairs = make_counterfactuals(windows_path, output_dir, negative_types=("zero", "shuffle"), seed=0)

    assert (output_dir / "simple_pairs.npz").exists()
    assert (output_dir / "index.json").exists()
    with np.load(output_dir / "simple_pairs.npz") as loaded:
        assert loaded["positive_index"].shape == pairs["positive_index"].shape


def test_make_counterfactuals_module_cli(tmp_path):
    windows_dir = tmp_path / "windows"
    windows_dir.mkdir()
    with (windows_dir / "windows.jsonl").open("w", encoding="utf-8") as handle:
        for window in _windows():
            handle.write(json.dumps(window) + "\n")
    output_dir = tmp_path / "counterfactuals"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ppwam.make_counterfactuals",
            "--windows",
            str(windows_dir),
            "--output",
            str(output_dir),
            "--types",
            "zero,shuffle",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "wrote" in result.stdout
    assert (output_dir / "simple_pairs.npz").exists()
