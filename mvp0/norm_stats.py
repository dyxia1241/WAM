from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


STAT_KEYS = ("proprio", "action")


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return loaded


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def train_episode_ids(windows_dir: str | Path, split: str = "train") -> list[str]:
    index_path = Path(windows_dir) / "index.json"
    index = read_json(index_path)
    split_map = index.get("split", {})
    if not isinstance(split_map, dict):
        raise ValueError(f"split must be an object in {index_path}.")
    episodes = split_map.get(split, [])
    if not isinstance(episodes, list) or not episodes:
        raise ValueError(f"No episodes found for split={split} in {index_path}.")
    return [str(episode_id) for episode_id in episodes]


def _init_accumulator(dim: int) -> dict[str, np.ndarray | int]:
    return {
        "count": 0,
        "sum": np.zeros(dim, dtype=np.float64),
        "sumsq": np.zeros(dim, dtype=np.float64),
        "min": np.full(dim, np.inf, dtype=np.float64),
        "max": np.full(dim, -np.inf, dtype=np.float64),
    }


def _update_accumulator(accumulator: dict[str, np.ndarray | int], values: np.ndarray) -> None:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("Expected arrays with shape [T, D].")
    accumulator["count"] = int(accumulator["count"]) + int(values.shape[0])
    accumulator["sum"] = np.asarray(accumulator["sum"]) + values.sum(axis=0)
    accumulator["sumsq"] = np.asarray(accumulator["sumsq"]) + np.square(values).sum(axis=0)
    accumulator["min"] = np.minimum(np.asarray(accumulator["min"]), values.min(axis=0))
    accumulator["max"] = np.maximum(np.asarray(accumulator["max"]), values.max(axis=0))


def _finalize_accumulator(accumulator: dict[str, np.ndarray | int], eps: float) -> dict[str, Any]:
    count = int(accumulator["count"])
    if count <= 0:
        raise ValueError("Cannot finalize empty normalization stats.")
    total = np.asarray(accumulator["sum"], dtype=np.float64)
    sumsq = np.asarray(accumulator["sumsq"], dtype=np.float64)
    mean = total / count
    var = np.maximum(sumsq / count - np.square(mean), 0.0)
    std = np.maximum(np.sqrt(var), eps)
    return {
        "count": count,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": np.asarray(accumulator["min"], dtype=np.float64).tolist(),
        "max": np.asarray(accumulator["max"], dtype=np.float64).tolist(),
        "eps": float(eps),
    }


def compute_norm_stats(
    windows_dir: str | Path,
    episodes_dir: str | Path,
    output: str | Path | None = None,
    split: str = "train",
    eps: float = 1.0e-6,
) -> dict[str, Any]:
    windows_dir = Path(windows_dir)
    episodes_dir = Path(episodes_dir)
    episode_ids = train_episode_ids(windows_dir, split=split)

    accumulators: dict[str, dict[str, np.ndarray | int]] = {}
    frame_count = 0
    for episode_id in episode_ids:
        arrays_path = episodes_dir / episode_id / "arrays.npz"
        if not arrays_path.exists():
            raise FileNotFoundError(arrays_path)
        with np.load(arrays_path) as arrays:
            for key in STAT_KEYS:
                if key not in arrays:
                    raise ValueError(f"{arrays_path} must contain {key}.")
                values = arrays[key]
                if key not in accumulators:
                    accumulators[key] = _init_accumulator(int(values.shape[1]))
                _update_accumulator(accumulators[key], values)
            frame_count += int(arrays["action"].shape[0])

    payload: dict[str, Any] = {
        "mode": "zscore",
        "source": {
            "windows_dir": str(windows_dir),
            "episodes_dir": str(episodes_dir),
            "split": split,
        },
        "counts": {
            "episodes": len(episode_ids),
            "frames": frame_count,
        },
    }
    for key in STAT_KEYS:
        payload[key] = _finalize_accumulator(accumulators[key], eps=eps)

    if output is not None:
        write_json(output, payload)
    return payload


def load_norm_stats(path_or_stats: str | Path | dict[str, Any]) -> dict[str, Any]:
    stats = read_json(path_or_stats) if isinstance(path_or_stats, (str, Path)) else dict(path_or_stats)
    mode = stats.get("mode", "zscore")
    if mode != "zscore":
        raise ValueError(f"Unsupported normalization mode: {mode}")
    for key in STAT_KEYS:
        if key not in stats:
            raise ValueError(f"Normalization stats missing {key}.")
        for field in ("mean", "std"):
            if field not in stats[key]:
                raise ValueError(f"Normalization stats missing {key}.{field}.")
    return stats


def normalize_array(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return ((values - mean) / std).astype(np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build train-split joint normalization stats.")
    parser.add_argument("--windows", required=True, help="Prepared windows directory.")
    parser.add_argument("--episodes", required=True, help="WAM episode directory.")
    parser.add_argument("--output", required=True, help="Output norm_stats.json path.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--eps", type=float, default=1.0e-6)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = compute_norm_stats(
        windows_dir=args.windows,
        episodes_dir=args.episodes,
        output=args.output,
        split=args.split,
        eps=args.eps,
    )
    print(json.dumps({"output": args.output, "counts": stats["counts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
