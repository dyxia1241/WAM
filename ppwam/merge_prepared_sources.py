from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ppwam.labels import STAGE_TO_ID
from ppwam.prepare_windows import read_episode_meta


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SourceSpec:
    name: str
    windows_dir: Path
    episodes_dir: Path
    features_dir: Path
    prompt_features: Path | None = None
    norm_stats: Path | None = None


def read_windows(path: str | Path) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected JSON object lines in {path}.")
                windows.append(row)
    if not windows:
        raise ValueError(f"No windows found in {path}.")
    return windows


def parse_optional_path(value: str) -> Path | None:
    value = value.strip()
    if not value or value == "-":
        return None
    return Path(value)


def parse_source_spec(text: str) -> SourceSpec:
    if "=" not in text:
        raise ValueError("Source specs must use name=windows,episodes,features,prompt,norm.")
    name, raw = text.split("=", 1)
    parts = [item.strip() for item in raw.split(",")]
    if len(parts) not in {3, 4, 5}:
        raise ValueError("Source specs must have 3 to 5 comma-separated paths after name=.")
    if not name.strip():
        raise ValueError("Source name must not be empty.")
    while len(parts) < 5:
        parts.append("-")
    return SourceSpec(
        name=name.strip(),
        windows_dir=Path(parts[0]),
        episodes_dir=Path(parts[1]),
        features_dir=Path(parts[2]),
        prompt_features=parse_optional_path(parts[3]),
        norm_stats=parse_optional_path(parts[4]),
    )


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


def ensure_output_safe(output_dir: Path, overwrite: bool) -> None:
    managed = ("windows.jsonl", "labels.npz", "index.json")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        return
    existing = [output_dir / name for name in managed if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{output_dir} already contains merged outputs. Use --overwrite to replace them.")
    if overwrite:
        for path in existing:
            path.unlink()


def source_window_records(source: SourceSpec, source_id: int) -> list[dict[str, Any]]:
    windows = read_windows(source.windows_dir / "windows.jsonl")
    out: list[dict[str, Any]] = []
    for source_index, row in enumerate(windows):
        raw_task_id = str(row["task_id"])
        record = dict(row)
        record["source"] = source.name
        record["source_id"] = int(source_id)
        record["raw_task_id"] = raw_task_id
        record["source_window_id"] = str(row.get("window_id", f"{row['episode_id']}_t{int(row['t']):06d}"))
        record["source_window_index"] = int(source_index)
        record["task_id"] = f"{source.name}::{raw_task_id}"
        record["window_id"] = f"{source.name}::{record['source_window_id']}"
        out.append(record)
    return out


def sample_equal_by_split(
    by_source: dict[str, list[dict[str, Any]]],
    caps: dict[str, int],
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {
        split: {source: [] for source in by_source} for split in SPLITS
    }
    for source, records in by_source.items():
        for record in records:
            split = str(record["split"])
            if split not in grouped:
                continue
            grouped[split][source].append(record)

    selected: list[dict[str, Any]] = []
    counts: dict[str, dict[str, int]] = {split: {} for split in SPLITS}
    for split in SPLITS:
        source_counts = {source: len(records) for source, records in grouped[split].items()}
        if any(count <= 0 for count in source_counts.values()):
            raise ValueError(f"Cannot equalize split={split}; source counts={source_counts}.")
        target = min(source_counts.values())
        cap = int(caps.get(split, 0))
        if cap > 0:
            target = min(target, cap)
        if target <= 0:
            raise ValueError(f"Target window count for split={split} must be positive.")
        for source, records in grouped[split].items():
            rng = np.random.default_rng(stable_seed(seed, split, source))
            if len(records) == target:
                chosen = list(records)
            else:
                indices = np.sort(rng.choice(len(records), size=target, replace=False))
                chosen = [records[int(index)] for index in indices]
            selected.extend(chosen)
            counts[split][source] = len(chosen)
    selected.sort(key=lambda item: (str(item["split"]), int(item["source_id"]), int(item["source_window_index"])))
    return selected, counts


def source_dims(source: SourceSpec, records: list[dict[str, Any]]) -> tuple[int, int]:
    action_dim = 0
    proprio_dim = 0
    for episode_id in sorted({str(record["episode_id"]) for record in records}):
        meta = read_episode_meta(source.episodes_dir / episode_id, validate_arrays=False)
        action_dim = max(action_dim, int(meta.action_dim))
        proprio_dim = max(proprio_dim, int(meta.proprio_dim))
    if action_dim <= 0 or proprio_dim <= 0:
        raise ValueError(f"Could not infer dims for source={source.name}.")
    return action_dim, proprio_dim


def even_dim(value: int) -> int:
    return int(value) if int(value) % 2 == 0 else int(value) + 1


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_labels(output_dir: Path, records: list[dict[str, Any]], task_to_id: dict[str, int]) -> None:
    np.savez_compressed(
        output_dir / "labels.npz",
        delta_phi=np.asarray([float(record["delta_phi"]) for record in records], dtype=np.float32),
        stage_id=np.asarray([int(record["stage_id"]) for record in records], dtype=np.int64),
        task_id=np.asarray([task_to_id[str(record["task_id"])] for record in records], dtype=np.int64),
        source_id=np.asarray([int(record["source_id"]) for record in records], dtype=np.int64),
        primitive_time=np.asarray([float(record["primitive_time"]) for record in records], dtype=np.float32),
        is_success=np.asarray([bool(record["is_success"]) for record in records], dtype=np.bool_),
        cross_boundary=np.asarray([bool(record["cross_boundary"]) for record in records], dtype=np.bool_),
    )


def prompt_feature_rows(path: Path) -> tuple[list[str], np.ndarray]:
    with np.load(path) as loaded:
        if "task_ids" not in loaded or "features" not in loaded:
            raise ValueError(f"{path} must contain task_ids and features arrays.")
        task_ids = [str(item) for item in loaded["task_ids"].tolist()]
        features = loaded["features"].astype(np.float32)
    if features.ndim != 2 or len(task_ids) != int(features.shape[0]):
        raise ValueError(f"Invalid prompt feature store: {path}")
    return task_ids, features


def write_merged_prompt_features(path: Path, sources: list[SourceSpec]) -> dict[str, Any]:
    task_ids_out: list[str] = []
    features_out: list[np.ndarray] = []
    feature_dim: int | None = None
    for source in sources:
        if source.prompt_features is None:
            raise ValueError(f"Missing prompt feature path for source={source.name}.")
        task_ids, features = prompt_feature_rows(source.prompt_features)
        if feature_dim is None:
            feature_dim = int(features.shape[1])
        if int(features.shape[1]) != feature_dim:
            raise ValueError("All prompt feature stores must have the same feature dimension.")
        for task_id, feature in zip(task_ids, features, strict=True):
            task_ids_out.append(f"{source.name}::{task_id}")
            features_out.append(feature.astype(np.float32))
    if len(set(task_ids_out)) != len(task_ids_out):
        raise ValueError("Merged prompt features would contain duplicate task ids.")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, task_ids=np.asarray(task_ids_out), features=np.stack(features_out, axis=0))
    return {"prompt_features": str(path), "num_prompts": len(task_ids_out), "feature_dim": int(feature_dim or 0)}


def merge_prepared_sources(
    sources: list[SourceSpec],
    output_dir: str | Path,
    seed: int = 42,
    train_cap: int = 20_000,
    val_cap: int = 2_500,
    test_cap: int = 2_500,
    prompt_output: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if len(sources) < 2:
        raise ValueError("At least two sources are required.")
    names = [source.name for source in sources]
    if len(set(names)) != len(names):
        raise ValueError("Source names must be unique.")
    output_path = Path(output_dir)
    ensure_output_safe(output_path, overwrite=overwrite)

    source_to_id = {source.name: idx for idx, source in enumerate(sources)}
    by_source = {source.name: source_window_records(source, source_to_id[source.name]) for source in sources}
    caps = {"train": int(train_cap), "val": int(val_cap), "test": int(test_cap)}
    records, counts = sample_equal_by_split(by_source, caps=caps, seed=seed)
    task_to_id = {task_id: idx for idx, task_id in enumerate(sorted({str(record["task_id"]) for record in records}))}
    source_dim_map = {}
    for source in sources:
        action_dim, proprio_dim = source_dims(source, by_source[source.name])
        source_dim_map[source.name] = {"action_dim": action_dim, "proprio_dim": proprio_dim}
    canonical_action_dim = even_dim(max(int(item["action_dim"]) for item in source_dim_map.values()))
    canonical_proprio_dim = even_dim(max(int(item["proprio_dim"]) for item in source_dim_map.values()))

    write_jsonl(output_path / "windows.jsonl", records)
    write_labels(output_path, records, task_to_id=task_to_id)
    prompt_manifest = None
    if prompt_output is not None:
        prompt_path = Path(prompt_output)
        if prompt_path.exists() and not overwrite:
            raise FileExistsError(f"{prompt_path} exists. Use --overwrite to replace it.")
        prompt_manifest = write_merged_prompt_features(prompt_path, sources)

    split_episode_ids: dict[str, list[str]] = {}
    for split in SPLITS:
        ids = sorted({f"{record['source']}::{record['episode_id']}" for record in records if record["split"] == split})
        split_episode_ids[split] = ids
    index = {
        "num_windows": len(records),
        "split": split_episode_ids,
        "task_to_id": task_to_id,
        "source_to_id": source_to_id,
        "stage_to_id": STAGE_TO_ID,
        "sources": {
            source.name: {
                "windows_dir": str(source.windows_dir),
                "episodes_dir": str(source.episodes_dir),
                "features_dir": str(source.features_dir),
                "prompt_features": str(source.prompt_features) if source.prompt_features is not None else "",
                "norm_stats": str(source.norm_stats) if source.norm_stats is not None else "",
                **source_dim_map[source.name],
            }
            for source in sources
        },
        "counts_by_split_source": counts,
        "prompt_manifest": prompt_manifest,
        "params": {
            "seed": int(seed),
            "balance_unit": "prepared_windows",
            "split_caps": caps,
            "canonical_action_dim": int(canonical_action_dim),
            "canonical_proprio_dim": int(canonical_proprio_dim),
        },
    }
    with (output_path / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)
    return index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge prepared PP-WAM datasets with equal windows per source.")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Repeatable: name=windows_dir,episodes_dir,features_dir,prompt_features,norm_stats. Use '-' for optional paths.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-cap", type=int, default=20_000)
    parser.add_argument("--val-cap", type=int, default=2_500)
    parser.add_argument("--test-cap", type=int, default=2_500)
    parser.add_argument("--prompt-output", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    sources = [parse_source_spec(text) for text in args.source]
    index = merge_prepared_sources(
        sources=sources,
        output_dir=args.output,
        seed=args.seed,
        train_cap=args.train_cap,
        val_cap=args.val_cap,
        test_cap=args.test_cap,
        prompt_output=args.prompt_output,
        overwrite=args.overwrite,
    )
    print(json.dumps({"output": args.output, "num_windows": index["num_windows"], "counts": index["counts_by_split_source"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
