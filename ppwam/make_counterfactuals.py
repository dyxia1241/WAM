from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_NEGATIVE_TYPES = ("zero", "reverse", "shuffle", "wrong_arm", "scaled_0.25", "scaled_1.75")


def read_windows(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.is_dir():
        path = path / "windows.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    windows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                windows.append(json.loads(line))
    if not windows:
        raise ValueError(f"No windows found in {path}.")
    return windows


def build_shuffle_candidates(windows: list[dict[str, Any]]) -> dict[tuple[str, str], list[int]]:
    candidates: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, window in enumerate(windows):
        candidates[(str(window["split"]), str(window["stage"]))].append(idx)
    return dict(candidates)


def generate_simple_pairs(
    windows: list[dict[str, Any]],
    negative_types: tuple[str, ...] = DEFAULT_NEGATIVE_TYPES,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    shuffle_candidates = build_shuffle_candidates(windows)

    positive_indices: list[int] = []
    replacement_indices: list[int] = []
    negative_kinds: list[str] = []
    splits: list[str] = []

    for idx, window in enumerate(windows):
        split = str(window["split"])
        stage = str(window["stage"])
        for negative_type in negative_types:
            replacement = -1
            if negative_type == "shuffle":
                candidates = [item for item in shuffle_candidates.get((split, stage), []) if item != idx]
                if not candidates:
                    continue
                replacement = int(rng.choice(candidates))

            positive_indices.append(idx)
            replacement_indices.append(replacement)
            negative_kinds.append(negative_type)
            splits.append(split)

    return {
        "positive_index": np.asarray(positive_indices, dtype=np.int64),
        "replacement_index": np.asarray(replacement_indices, dtype=np.int64),
        "negative_kind": np.asarray(negative_kinds),
        "split": np.asarray(splits),
    }


def write_pairs(pairs: dict[str, np.ndarray], output_dir: str | Path, negative_types: tuple[str, ...], seed: int) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "simple_pairs.npz", **pairs)
    counts = Counter(str(item) for item in pairs["negative_kind"].tolist())
    index = {
        "num_pairs": int(len(pairs["positive_index"])),
        "counts_by_type": dict(sorted(counts.items())),
        "negative_types": list(negative_types),
        "seed": seed,
    }
    with (output_dir / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)


def make_counterfactuals(
    windows: str | Path,
    output: str | Path,
    negative_types: tuple[str, ...] = DEFAULT_NEGATIVE_TYPES,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    window_records = read_windows(windows)
    pairs = generate_simple_pairs(window_records, negative_types=negative_types, seed=seed)
    write_pairs(pairs, output, negative_types=negative_types, seed=seed)
    return pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate simple counterfactual pair indices.")
    parser.add_argument("--windows", required=True, help="Prepared windows dir or windows.jsonl path.")
    parser.add_argument("--output", required=True, help="Output directory for counterfactual pair index.")
    parser.add_argument(
        "--types",
        default=",".join(DEFAULT_NEGATIVE_TYPES),
        help="Comma-separated simple negative types.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    negative_types = tuple(item.strip() for item in args.types.split(",") if item.strip())
    pairs = make_counterfactuals(
        windows=args.windows,
        output=args.output,
        negative_types=negative_types,
        seed=args.seed,
    )
    print(f"wrote {len(pairs['positive_index'])} pairs to {args.output}")


if __name__ == "__main__":
    main()
