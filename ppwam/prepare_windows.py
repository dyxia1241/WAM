from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from ppwam.labels import STAGE_TO_ID, PrimitiveBoundary
from ppwam.schemas import EpisodeMeta, EpisodeSpec, WindowRecord
from ppwam.window_builder import build_windows, episode_to_split, split_episodes


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return loaded


def read_episode_spec(episode_dir: str | Path) -> EpisodeSpec:
    episode_dir = Path(episode_dir)
    meta = read_episode_meta(episode_dir, validate_arrays=True)
    labels_json = read_json(episode_dir / "labels.json")

    boundaries = tuple(
        PrimitiveBoundary(
            stage=str(item["stage"]),
            start=int(item["start"]),
            end=int(item["end"]),
        )
        for item in labels_json["primitive_boundaries"]
    )
    raw_potential = labels_json.get("potential", labels_json.get("phi"))
    potential = None
    if raw_potential is not None:
        if not isinstance(raw_potential, list):
            raise ValueError(f"potential/phi in {episode_dir / 'labels.json'} must be a list.")
        if len(raw_potential) != int(meta.num_frames):
            raise ValueError(
                f"potential/phi length for {meta.episode_id} must match num_frames={meta.num_frames}."
            )
        potential = tuple(float(item) for item in raw_potential)
    success = bool(labels_json.get("success", meta.success))
    meta = EpisodeMeta(
        **{**meta.__dict__, "success": success},
    )

    return EpisodeSpec(meta=meta, boundaries=boundaries, potential=potential)


def read_episode_meta(episode_dir: str | Path, validate_arrays: bool = False) -> EpisodeMeta:
    episode_dir = Path(episode_dir)
    meta_json = read_json(episode_dir / "meta.json")
    meta = EpisodeMeta(
        episode_id=str(meta_json["episode_id"]),
        task_id=str(meta_json["task_id"]),
        source=str(meta_json.get("source", "")),
        language=str(meta_json.get("language", "")),
        fps=int(meta_json.get("fps", 10)),
        num_frames=int(meta_json["num_frames"]),
        cameras=tuple(meta_json.get("cameras", ("cam0",))),
        action_dim=int(meta_json["action_dim"]),
        proprio_dim=int(meta_json["proprio_dim"]),
        success=bool(meta_json.get("success", True)),
    )

    if validate_arrays:
        arrays_path = episode_dir / "arrays.npz"
        if arrays_path.exists():
            with np.load(arrays_path) as arrays:
                if "proprio" not in arrays or "action" not in arrays:
                    raise ValueError(f"{arrays_path} must contain proprio and action arrays.")
                if arrays["proprio"].shape[0] != meta.num_frames:
                    raise ValueError(f"proprio frame count does not match meta for {meta.episode_id}.")
                if arrays["action"].shape[0] != meta.num_frames:
                    raise ValueError(f"action frame count does not match meta for {meta.episode_id}.")

    return meta


def read_episode_metas(episodes_root: str | Path, validate_arrays: bool = False) -> list[EpisodeMeta]:
    episodes_root = Path(episodes_root)
    if not episodes_root.exists():
        raise FileNotFoundError(episodes_root)
    episode_dirs = sorted(path for path in episodes_root.iterdir() if path.is_dir())
    if not episode_dirs:
        raise ValueError(f"No episode directories found in {episodes_root}.")
    return [read_episode_meta(path, validate_arrays=validate_arrays) for path in episode_dirs]


def read_episode_specs(episodes_root: str | Path, skip_missing_labels: bool = False) -> list[EpisodeSpec]:
    episodes_root = Path(episodes_root)
    if not episodes_root.exists():
        raise FileNotFoundError(episodes_root)
    episode_dirs = sorted(path for path in episodes_root.iterdir() if path.is_dir())
    if not episode_dirs:
        raise ValueError(f"No episode directories found in {episodes_root}.")
    specs: list[EpisodeSpec] = []
    for path in episode_dirs:
        if skip_missing_labels and not (path / "labels.json").exists():
            continue
        specs.append(read_episode_spec(path))
    if not specs:
        raise ValueError(f"No labeled episode directories found in {episodes_root}.")
    return specs


def window_record_to_json(record: WindowRecord) -> dict[str, Any]:
    payload = asdict(record)
    payload["history_indices"] = list(record.history_indices)
    payload["future_indices"] = list(record.future_indices)
    return payload


def write_prepared_windows(
    records: list[WindowRecord],
    output_dir: str | Path,
    split: dict[str, list[str]],
    task_to_id: dict[str, int],
    params: dict[str, Any],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_to_id = {source: idx for idx, source in enumerate(sorted({record.source for record in records}))}
    with (output_dir / "windows.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            payload = window_record_to_json(record)
            payload["source_id"] = source_to_id[record.source]
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    task_ids = np.asarray([task_to_id[record.task_id] for record in records], dtype=np.int64)
    source_ids = np.asarray([source_to_id[record.source] for record in records], dtype=np.int64)
    np.savez_compressed(
        output_dir / "labels.npz",
        delta_phi=np.asarray([record.delta_phi for record in records], dtype=np.float32),
        stage_id=np.asarray([record.stage_id for record in records], dtype=np.int64),
        task_id=task_ids,
        source_id=source_ids,
        primitive_time=np.asarray([record.primitive_time for record in records], dtype=np.float32),
        phi_t=np.asarray([record.phi_t for record in records], dtype=np.float32),
        phi_future=np.asarray([record.phi_future for record in records], dtype=np.float32),
        delta_phi_raw=np.asarray([record.delta_phi_raw for record in records], dtype=np.float32),
        is_success=np.asarray([record.is_success for record in records], dtype=np.bool_),
        cross_boundary=np.asarray([record.cross_boundary for record in records], dtype=np.bool_),
    )

    index = {
        "num_windows": len(records),
        "split": split,
        "task_to_id": task_to_id,
        "source_to_id": source_to_id,
        "stage_to_id": STAGE_TO_ID,
        "params": params,
    }
    with (output_dir / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)


def prepare_windows(
    episodes_root: str | Path,
    output_dir: str | Path,
    history: int = 4,
    horizon: int = 8,
    stride: int = 2,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    exclude_cross_boundary: bool = False,
    skip_missing_labels: bool = False,
) -> list[WindowRecord]:
    episodes = read_episode_specs(episodes_root, skip_missing_labels=skip_missing_labels)
    episode_ids = [episode.meta.episode_id for episode in episodes]
    split = split_episodes(episode_ids, ratios=(train_ratio, val_ratio, test_ratio), seed=seed)
    split_by_episode = episode_to_split(split)
    records = build_windows(
        episodes,
        split_by_episode=split_by_episode,
        history=history,
        horizon=horizon,
        stride=stride,
        exclude_cross_boundary=exclude_cross_boundary,
    )
    task_to_id = {task_id: idx for idx, task_id in enumerate(sorted({episode.meta.task_id for episode in episodes}))}
    write_prepared_windows(
        records,
        output_dir=output_dir,
        split=split,
        task_to_id=task_to_id,
        params={
            "history": history,
            "horizon": horizon,
            "stride": stride,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "seed": seed,
            "exclude_cross_boundary": exclude_cross_boundary,
            "skip_missing_labels": skip_missing_labels,
        },
    )
    return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare PP-WAM windows from episode directories.")
    parser.add_argument("--episodes", required=True, help="Path to data/episodes.")
    parser.add_argument("--output", required=True, help="Output path for prepared windows.")
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-cross-boundary", action="store_true")
    parser.add_argument("--skip-missing-labels", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = prepare_windows(
        episodes_root=args.episodes,
        output_dir=args.output,
        history=args.history,
        horizon=args.horizon,
        stride=args.stride,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        exclude_cross_boundary=args.exclude_cross_boundary,
        skip_missing_labels=args.skip_missing_labels,
    )
    print(f"wrote {len(records)} windows to {args.output}")


if __name__ == "__main__":
    main()
