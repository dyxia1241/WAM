from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


PROPRIO_COLUMNS = (
    "observation.state.arm.position",
    "observation.state.effector.position",
)
ACTION_COLUMNS = (
    "action.arm.position",
    "action.effector.position",
)
CAMERA_PREFIX = "observation.images."


@dataclass(frozen=True)
class GM100EpisodeSelection:
    task_id: str
    episode_id: str


@dataclass(frozen=True)
class ImportedEpisode:
    episode_id: str
    task_id: str
    source_episode_id: str
    num_frames: int
    cameras: tuple[str, ...]
    output_dir: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import a raw GM-100 subset into WAM episode layout.")
    parser.add_argument("--raw-root", required=True, help="Raw GM-100 subset directory.")
    parser.add_argument("--output", required=True, help="Output directory for WAM episode folders.")
    parser.add_argument("--manifest", default=None, help="Defaults to <raw-root>/gm100_subset_manifest.json.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-frames", type=int, default=None, help="Optional small smoke-test frame cap.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return loaded


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            loaded = json.loads(line)
            if not isinstance(loaded, dict):
                raise ValueError(f"Expected JSON object lines in {path}.")
            rows.append(loaded)
    return rows


def read_subset_manifest(raw_root: str | Path, manifest: str | Path | None = None) -> dict[str, Any]:
    path = Path(manifest) if manifest is not None else Path(raw_root) / "gm100_subset_manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return read_json(path)


def selected_episodes_from_manifest(manifest: dict[str, Any]) -> list[GM100EpisodeSelection]:
    selected = manifest.get("selected_episodes")
    if not isinstance(selected, list) or not selected:
        raise ValueError("Manifest must contain a non-empty selected_episodes list.")
    episodes: list[GM100EpisodeSelection] = []
    for item in selected:
        if not isinstance(item, dict):
            raise ValueError("selected_episodes entries must be objects.")
        episodes.append(
            GM100EpisodeSelection(
                task_id=str(item["task_id"]),
                episode_id=str(item["episode_id"]),
            )
        )
    return episodes


def episode_index(episode_id: str) -> int:
    prefix = "episode_"
    if not episode_id.startswith(prefix):
        raise ValueError(f"Unexpected episode id: {episode_id}")
    return int(episode_id[len(prefix) :])


def wam_episode_id(task_id: str, episode_id: str) -> str:
    return f"{task_id}_{episode_id}"


def camera_feature_to_name(feature_name: str) -> str:
    if not feature_name.startswith(CAMERA_PREFIX):
        raise ValueError(f"Not a GM-100 camera feature: {feature_name}")
    return feature_name[len(CAMERA_PREFIX) :]


def camera_name_to_feature(camera_name: str) -> str:
    return f"{CAMERA_PREFIX}{camera_name}"


def camera_features_from_info(info: dict[str, Any]) -> list[str]:
    features = info.get("features", {})
    if not isinstance(features, dict):
        raise ValueError("info.json must contain a features object.")
    cameras = [
        name
        for name, spec in features.items()
        if name.startswith(CAMERA_PREFIX) and isinstance(spec, dict) and spec.get("dtype") == "video"
    ]
    if not cameras:
        raise ValueError("No video camera features found in info.json.")
    return sorted(cameras)


def task_language(raw_root: Path, task_id: str, episode_id: str) -> str:
    episode_rows = read_jsonl(raw_root / task_id / "meta" / "episodes.jsonl")
    target_index = episode_index(episode_id)
    for row in episode_rows:
        if int(row.get("episode_index", -1)) == target_index:
            tasks = row.get("tasks", [])
            if tasks:
                return str(tasks[0])

    task_rows = read_jsonl(raw_root / task_id / "meta" / "tasks.jsonl")
    if not task_rows:
        return ""
    return str(task_rows[0].get("task", ""))


def episode_length(raw_root: Path, task_id: str, episode_id: str) -> int:
    rows = read_jsonl(raw_root / task_id / "meta" / "episodes.jsonl")
    target_index = episode_index(episode_id)
    for row in rows:
        if int(row.get("episode_index", -1)) == target_index:
            return int(row["length"])
    raise ValueError(f"{task_id}/{episode_id} not found in episodes.jsonl.")


def parquet_path(raw_root: Path, selection: GM100EpisodeSelection) -> Path:
    return raw_root / selection.task_id / "data" / "chunk-000" / f"{selection.episode_id}.parquet"


def video_path(raw_root: Path, selection: GM100EpisodeSelection, camera_feature: str) -> Path:
    return raw_root / selection.task_id / "videos" / "chunk-000" / camera_feature / f"{selection.episode_id}.mp4"


def require_pyarrow_parquet():
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyarrow is required to import GM-100 parquet files.") from exc
    return pq


def require_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("opencv-python-headless is required to extract GM-100 video frames.") from exc
    return cv2


def column_to_array(table, column: str, dtype: np.dtype | type = np.float32) -> np.ndarray:
    if column not in table.column_names:
        raise ValueError(f"Missing parquet column: {column}")
    values = table[column].to_pylist()
    array = np.asarray(values, dtype=dtype)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Column {column} must convert to a 2D array; got shape {array.shape}.")
    return array


def read_parquet_arrays(path: str | Path) -> dict[str, np.ndarray]:
    pq = require_pyarrow_parquet()
    table = pq.read_table(path)
    proprio = np.concatenate([column_to_array(table, column) for column in PROPRIO_COLUMNS], axis=1)
    action = np.concatenate([column_to_array(table, column) for column in ACTION_COLUMNS], axis=1)
    arrays: dict[str, np.ndarray] = {
        "proprio": proprio.astype(np.float32),
        "action": action.astype(np.float32),
    }
    for column, dtype in (("timestamp", np.float32), ("frame_index", np.int64), ("episode_index", np.int64)):
        if column in table.column_names:
            arrays[column] = column_to_array(table, column, dtype=dtype).squeeze(axis=1)
    return arrays


def slice_arrays(arrays: dict[str, np.ndarray], num_frames: int) -> dict[str, np.ndarray]:
    return {key: value[:num_frames].copy() for key, value in arrays.items()}


def clear_image_dir(path: Path) -> None:
    if not path.exists():
        return
    for image in path.iterdir():
        if image.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image.unlink()


def extract_video_frames(
    video: str | Path,
    output_dir: str | Path,
    expected_frames: int,
    jpeg_quality: int,
    overwrite: bool,
) -> int:
    cv2 = require_cv2()
    output_dir = Path(output_dir)
    existing = sorted(output_dir.glob("*.jpg"))
    if existing and not overwrite:
        if len(existing) != expected_frames:
            raise ValueError(f"{output_dir} has {len(existing)} images; expected {expected_frames}.")
        return len(existing)

    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        clear_image_dir(output_dir)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video}")
    try:
        for index in range(expected_frames):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"{video} ended at frame {index}; expected {expected_frames}.")
            output_path = output_dir / f"{index:06d}.jpg"
            wrote = cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            if not wrote:
                raise ValueError(f"Failed to write {output_path}")
    finally:
        capture.release()
    return expected_frames


def import_one_episode(
    raw_root: Path,
    output_root: Path,
    selection: GM100EpisodeSelection,
    jpeg_quality: int,
    max_frames: int | None,
    overwrite: bool,
) -> ImportedEpisode:
    info = read_json(raw_root / selection.task_id / "meta" / "info.json")
    cameras = camera_features_from_info(info)
    arrays = read_parquet_arrays(parquet_path(raw_root, selection))
    source_length = episode_length(raw_root, selection.task_id, selection.episode_id)
    if arrays["action"].shape[0] != arrays["proprio"].shape[0]:
        raise ValueError(f"Action/proprio frame mismatch in {selection.task_id}/{selection.episode_id}.")
    if arrays["action"].shape[0] != source_length:
        raise ValueError(
            f"Parquet has {arrays['action'].shape[0]} frames but episodes.jsonl reports {source_length} "
            f"for {selection.task_id}/{selection.episode_id}."
        )

    num_frames = source_length if max_frames is None else min(source_length, max_frames)
    arrays = slice_arrays(arrays, num_frames)
    episode_id = wam_episode_id(selection.task_id, selection.episode_id)
    episode_dir = output_root / episode_id
    if episode_dir.exists() and overwrite:
        shutil.rmtree(episode_dir)
    if episode_dir.exists() and not overwrite:
        raise FileExistsError(f"{episode_dir} exists. Pass --overwrite to replace it.")
    episode_dir.mkdir(parents=True, exist_ok=True)

    camera_names = tuple(camera_feature_to_name(camera) for camera in cameras)
    for camera_feature, camera_name in zip(cameras, camera_names, strict=True):
        extract_video_frames(
            video=video_path(raw_root, selection, camera_feature),
            output_dir=episode_dir / "images" / camera_name,
            expected_frames=num_frames,
            jpeg_quality=jpeg_quality,
            overwrite=overwrite,
        )

    meta = {
        "episode_id": episode_id,
        "task_id": selection.task_id,
        "source": "gm100",
        "source_episode_id": selection.episode_id,
        "language": task_language(raw_root, selection.task_id, selection.episode_id),
        "fps": int(info.get("fps", 30)),
        "num_frames": num_frames,
        "cameras": list(camera_names),
        "action_dim": int(arrays["action"].shape[1]),
        "proprio_dim": int(arrays["proprio"].shape[1]),
        "action_space": "absolute",
    }
    (episode_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(episode_dir / "arrays.npz", **arrays)
    import_manifest = {
        "source_task_id": selection.task_id,
        "source_episode_id": selection.episode_id,
        "source_length": source_length,
        "num_frames": num_frames,
        "camera_features": cameras,
        "proprio_columns": PROPRIO_COLUMNS,
        "action_columns": ACTION_COLUMNS,
    }
    (episode_dir / "import_manifest.json").write_text(
        json.dumps(import_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return ImportedEpisode(
        episode_id=episode_id,
        task_id=selection.task_id,
        source_episode_id=selection.episode_id,
        num_frames=num_frames,
        cameras=camera_names,
        output_dir=str(episode_dir),
    )


def import_gm100_subset(
    raw_root: str | Path,
    output: str | Path,
    manifest: str | Path | None = None,
    jpeg_quality: int = 95,
    max_frames: int | None = None,
    overwrite: bool = False,
) -> list[ImportedEpisode]:
    raw_root = Path(raw_root)
    output = Path(output)
    subset_manifest = read_subset_manifest(raw_root, manifest)
    selections = selected_episodes_from_manifest(subset_manifest)
    imported: list[ImportedEpisode] = []
    for selection in selections:
        imported.append(
            import_one_episode(
                raw_root=raw_root,
                output_root=output,
                selection=selection,
                jpeg_quality=jpeg_quality,
                max_frames=max_frames,
                overwrite=overwrite,
            )
        )

    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "raw_root": str(raw_root),
        "num_episodes": len(imported),
        "episodes": [asdict(item) for item in imported],
    }
    (output / "gm100_import_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return imported


def main() -> None:
    args = build_parser().parse_args()
    imported = import_gm100_subset(
        raw_root=args.raw_root,
        output=args.output,
        manifest=args.manifest,
        jpeg_quality=args.jpeg_quality,
        max_frames=args.max_frames,
        overwrite=args.overwrite,
    )
    print(f"imported {len(imported)} GM-100 episodes to {args.output}")


if __name__ == "__main__":
    main()
