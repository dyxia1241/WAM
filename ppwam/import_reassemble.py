from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ppwam.prompts import PromptRecord, encode_prompts_mock, format_prompt
from ppwam.prompts import write_prompt_feature_store, write_prompt_table


DEFAULT_CAMERA = "hand"
STAGE_VOCAB = ("approach", "grasp", "move", "release")
PROPRIO_LAYOUT = (
    "pose_0",
    "pose_1",
    "pose_2",
    "pose_3",
    "pose_4",
    "pose_5",
    "pose_6",
    "force_x",
    "force_y",
    "force_z",
    "gripper_position",
    "tcp_speed_ema",
)


@dataclass(frozen=True)
class ImportedReassembleEpisode:
    episode_id: str
    task_id: str
    recording_id: str
    num_frames: int
    num_boundaries: int
    output_dir: str


@dataclass(frozen=True)
class ReassembleSignals:
    timestamps: np.ndarray
    proprio: np.ndarray
    action: np.ndarray
    speed_ema: np.ndarray
    force_mag: np.ndarray


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).replace("\u3000", " ").strip().split())


def is_no_action(text: str) -> bool:
    return normalize_text(text).rstrip(".").lower() == "no action"


def high_level_text(segment: dict[str, Any]) -> str:
    return normalize_text(segment.get("text", "")).rstrip(".")


def object_from_high_level_text(text: str) -> str:
    text = normalize_text(text).rstrip(".")
    if not text or is_no_action(text):
        return ""
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else ""


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                loaded = json.loads(line)
                if not isinstance(loaded, dict):
                    raise ValueError(f"Expected JSON object lines in {path}.")
                rows.append(loaded)
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows


def recording_h5_path(dataset_root: str | Path, recording_id: str) -> Path:
    return Path(dataset_root) / "data" / f"{recording_id}.h5"


def filter_recordings(
    rows: Iterable[dict[str, Any]],
    dataset_root: str | Path,
    split: str,
    max_recordings: int = 0,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in rows:
        if normalize_text(row.get("split")) != split:
            continue
        recording_id = normalize_text(row.get("recording_id"))
        if not recording_id:
            continue
        if not recording_h5_path(dataset_root, recording_id).exists():
            continue
        selected.append(row)
        if max_recordings > 0 and len(selected) >= int(max_recordings):
            break
    if not selected:
        raise ValueError(f"No REASSEMBLE recordings selected for split={split}.")
    return selected


def estimate_fps(timestamps: np.ndarray) -> int:
    if timestamps.size < 2:
        return 30
    diffs = np.diff(timestamps.astype(np.float64))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 30
    median = float(np.median(diffs))
    if median > 10.0:
        median = median / 1000.0
    return max(1, int(round(1.0 / median)))


def ema1d(values: np.ndarray, span: int = 9) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"ema1d expects 1D input, got {arr.shape}.")
    if arr.size == 0:
        return arr.copy()
    out = np.zeros_like(arr, dtype=np.float64)
    alpha = 2.0 / float(span + 1)
    out[0] = float(arr[0])
    for idx in range(1, int(arr.size)):
        out[idx] = alpha * float(arr[idx]) + (1.0 - alpha) * out[idx - 1]
    return out


def nearest_timestamp_indices(source_timestamps: np.ndarray, query_timestamps: np.ndarray) -> np.ndarray:
    if source_timestamps.size == 0:
        raise ValueError("source_timestamps is empty.")
    idx = np.searchsorted(source_timestamps, query_timestamps, side="left")
    idx = np.clip(idx, 0, int(source_timestamps.size - 1))
    left = np.clip(idx - 1, 0, int(source_timestamps.size - 1))
    choose_left = np.abs(source_timestamps[left] - query_timestamps) <= np.abs(source_timestamps[idx] - query_timestamps)
    return np.where(choose_left, left, idx).astype(np.int64)


def h5_array(h5_file: Any, path: str) -> np.ndarray:
    if path not in h5_file:
        raise KeyError(path)
    return np.asarray(h5_file[path][:])


def stream_for_camera(h5_file: Any, key: str, camera_timestamps: np.ndarray) -> np.ndarray:
    values = h5_array(h5_file, f"robot_state/{key}")
    if values.ndim == 1:
        values = values[:, None]
    ts_path = f"timestamps/{key}"
    if ts_path in h5_file:
        timestamps = np.asarray(h5_file[ts_path][:], dtype=np.float64)
        indices = nearest_timestamp_indices(timestamps, camera_timestamps)
        return values[indices]
    if values.shape[0] == camera_timestamps.shape[0]:
        return values
    raise ValueError(f"Cannot align robot_state/{key}; missing {ts_path} and length differs from camera.")


def fixed_dim(values: np.ndarray, dim: int, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape [T,D], got {arr.shape}.")
    if arr.shape[1] >= dim:
        return arr[:, :dim]
    pad = np.zeros((arr.shape[0], dim - arr.shape[1]), dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=1)


def force_stream(h5_file: Any, camera_timestamps: np.ndarray) -> np.ndarray:
    for key in ("compensated_base_force", "measured_force"):
        if f"robot_state/{key}" in h5_file:
            return fixed_dim(stream_for_camera(h5_file, key, camera_timestamps), 3, key)
    raise ValueError("REASSEMBLE H5 is missing compensated_base_force/measured_force.")


def load_reassemble_signals(h5_path: str | Path, camera: str = DEFAULT_CAMERA) -> ReassembleSignals:
    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py is required to import REASSEMBLE H5 files.") from exc

    with h5py.File(h5_path, "r") as h5_file:
        camera_timestamps = np.asarray(h5_file[f"timestamps/{camera}"][:], dtype=np.float64)
        if camera_timestamps.size == 0:
            raise ValueError(f"No timestamps for camera={camera} in {h5_path}.")
        pose = fixed_dim(stream_for_camera(h5_file, "pose", camera_timestamps), 7, "pose")
        force_xyz = force_stream(h5_file, camera_timestamps)
        gripper = fixed_dim(stream_for_camera(h5_file, "gripper_positions", camera_timestamps), 1, "gripper_positions")
        velocity = fixed_dim(stream_for_camera(h5_file, "velocity", camera_timestamps), 3, "velocity")

    speed_ema = ema1d(np.linalg.norm(velocity[:, :3], axis=1), span=9)
    force_mag = np.linalg.norm(force_xyz[:, :3], axis=1)
    proprio = np.concatenate([pose[:, :7], force_xyz[:, :3], gripper[:, :1], speed_ema[:, None]], axis=1).astype(np.float32)
    action = np.concatenate([proprio[1:], proprio[-1:]], axis=0).astype(np.float32)
    return ReassembleSignals(
        timestamps=camera_timestamps.astype(np.float64),
        proprio=proprio,
        action=action,
        speed_ema=speed_ema.astype(np.float32),
        force_mag=force_mag.astype(np.float32),
    )


def timestamp_interval_to_frame_range(
    timestamps: np.ndarray,
    start_time: float,
    end_time: float,
) -> tuple[int, int] | None:
    if timestamps.size == 0:
        return None
    lo = int(np.searchsorted(timestamps, start_time, side="left"))
    hi = int(np.searchsorted(timestamps, end_time, side="right")) - 1
    lo = max(0, min(lo, int(timestamps.size - 1)))
    hi = max(0, min(hi, int(timestamps.size - 1)))
    if hi < lo:
        return None
    return lo, hi


def low_level_to_stage(low_text: str) -> str | None:
    low_text = normalize_text(low_text).title()
    if not low_text or low_text == "No Action":
        return None
    if low_text == "Approach":
        return "approach"
    if low_text in {"Align", "Grasp", "Nudge"}:
        return "grasp"
    if low_text in {"Lift", "Pull", "Push", "Twist"}:
        return "move"
    if low_text == "Release":
        return "release"
    return "move"


def build_primitive_boundaries(
    recording_row: dict[str, Any],
    timestamps: np.ndarray,
    min_stage_span: int = 10,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for segment in recording_row.get("segments", []):
        high_text = high_level_text(segment)
        if is_no_action(high_text) or not bool(segment.get("success", False)):
            continue
        segment_index = int(segment.get("segment_index", 0))
        for low_pos, low in enumerate(segment.get("low_level", [])):
            low_text = normalize_text(low.get("text", "")).rstrip(".")
            if not low_text or not bool(low.get("success", True)):
                continue
            stage = low_level_to_stage(low_text)
            if stage is None:
                continue
            frame_range = timestamp_interval_to_frame_range(timestamps, float(low.get("start", 0.0)), float(low.get("end", 0.0)))
            if frame_range is None:
                continue
            start, end = frame_range
            if end - start + 1 < int(min_stage_span):
                continue
            boundary = {"stage": stage, "start": int(start), "end": int(end)}
            raw.append(boundary)
            metadata.append(
                {
                    "segment_index": segment_index,
                    "low_index": int(low.get("low_index", low_pos)),
                    "stage": stage,
                    "low_level_text_raw": low_text,
                    "high_level_text_raw": high_text,
                    "object_raw": object_from_high_level_text(high_text),
                    "timestamp_start": float(low.get("start", 0.0)),
                    "timestamp_end": float(low.get("end", 0.0)),
                    "boundary_start": int(start),
                    "boundary_end": int(end),
                }
            )

    boundaries: list[dict[str, Any]] = []
    previous_end = -1
    for boundary in sorted(raw, key=lambda item: (int(item["start"]), int(item["end"]))):
        item = dict(boundary)
        if int(item["start"]) <= previous_end:
            item["start"] = previous_end + 1
        if int(item["end"]) - int(item["start"]) + 1 >= int(min_stage_span):
            boundaries.append(item)
            previous_end = int(item["end"])
    if not boundaries:
        raise ValueError("No valid REASSEMBLE primitive boundaries found.")
    return boundaries, metadata


def recording_language(recording_row: dict[str, Any]) -> str:
    texts: list[str] = []
    for text in recording_row.get("high_level_texts", []):
        cleaned = normalize_text(text).rstrip(".")
        if cleaned and not is_no_action(cleaned) and cleaned not in texts:
            texts.append(cleaned)
    return "; ".join(texts[:24])


def extract_video_frames_from_h5(
    h5_path: str | Path,
    camera: str,
    output_dir: str | Path,
    expected_frames: int,
    jpeg_quality: int,
    overwrite: bool,
) -> int:
    try:
        import cv2
        import h5py
    except ModuleNotFoundError as exc:
        raise RuntimeError("h5py and opencv-python-headless are required for --extract-images.") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("*.jpg"))
    if existing and not overwrite:
        if len(existing) != int(expected_frames):
            raise ValueError(f"{output_dir} has {len(existing)} images; expected {expected_frames}.")
        return len(existing)
    if overwrite:
        for image in output_dir.glob("*.jpg"):
            image.unlink()

    with h5py.File(h5_path, "r") as h5_file:
        if camera not in h5_file:
            raise KeyError(f"{camera} video payload not found in {h5_path}.")
        payload = bytes(np.asarray(h5_file[camera][()]).tobytes())

    with tempfile.NamedTemporaryFile(suffix=f"_{camera}.mp4", delete=True) as tmp:
        tmp.write(payload)
        tmp.flush()
        capture = cv2.VideoCapture(tmp.name)
        if not capture.isOpened():
            raise ValueError(f"Could not open embedded video for camera={camera}: {h5_path}")
        try:
            for index in range(int(expected_frames)):
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Embedded video ended at frame {index}; expected {expected_frames}.")
                path = output_dir / f"{index:06d}.jpg"
                wrote = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                if not wrote:
                    raise ValueError(f"Failed to write {path}.")
        finally:
            capture.release()
    return int(expected_frames)


def import_one_recording(
    dataset_root: Path,
    output_root: Path,
    recording_row: dict[str, Any],
    camera: str,
    min_stage_span: int,
    extract_images: bool,
    jpeg_quality: int,
    overwrite: bool,
) -> ImportedReassembleEpisode:
    recording_id = normalize_text(recording_row["recording_id"])
    h5_path = recording_h5_path(dataset_root, recording_id)
    signals = load_reassemble_signals(h5_path, camera=camera)
    boundaries, primitive_metadata = build_primitive_boundaries(
        recording_row,
        timestamps=signals.timestamps,
        min_stage_span=min_stage_span,
    )

    episode_id = recording_id
    episode_dir = output_root / episode_id
    if episode_dir.exists() and overwrite:
        shutil.rmtree(episode_dir)
    if episode_dir.exists() and not overwrite:
        raise FileExistsError(f"{episode_dir} exists. Pass --overwrite to replace it.")
    episode_dir.mkdir(parents=True, exist_ok=True)

    if extract_images:
        extract_video_frames_from_h5(
            h5_path=h5_path,
            camera=camera,
            output_dir=episode_dir / "images" / camera,
            expected_frames=int(signals.proprio.shape[0]),
            jpeg_quality=jpeg_quality,
            overwrite=overwrite,
        )

    language = recording_language(recording_row)
    meta = {
        "episode_id": episode_id,
        "task_id": recording_id,
        "source": "reassemble",
        "recording_id": recording_id,
        "language": language,
        "fps": estimate_fps(signals.timestamps),
        "num_frames": int(signals.proprio.shape[0]),
        "cameras": [camera],
        "action_dim": int(signals.action.shape[1]),
        "proprio_dim": int(signals.proprio.shape[1]),
        "action_space": "absolute_pose_force_proxy",
        "images_imported": bool(extract_images),
        "split": normalize_text(recording_row.get("split")),
        "n_success_high_segments": int(recording_row.get("n_success_high_segments", 0)),
        "n_failed_high_segments": int(recording_row.get("n_failed_high_segments", 0)),
    }
    (episode_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(
        episode_dir / "arrays.npz",
        proprio=signals.proprio,
        action=signals.action,
        timestamp=signals.timestamps.astype(np.float64),
        frame_index=np.arange(signals.proprio.shape[0], dtype=np.int64),
        force_mag=signals.force_mag.astype(np.float32),
        tcp_speed=signals.speed_ema.astype(np.float32),
    )
    labels = {
        "success": True,
        "label_source": "reassemble_low_level_segments_v1",
        "recording_id": recording_id,
        "params": {
            "camera": camera,
            "min_stage_span": int(min_stage_span),
            "stage_vocab": list(STAGE_VOCAB),
            "kept_segments": "successful_non_no_action",
        },
        "primitive_boundaries": boundaries,
        "primitive_metadata": primitive_metadata,
    }
    (episode_dir / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True), encoding="utf-8")
    (episode_dir / "import_manifest.json").write_text(
        json.dumps(
            {
                "dataset_root": str(dataset_root),
                "h5_path": str(h5_path),
                "recording_id": recording_id,
                "camera": camera,
                "num_frames": int(signals.proprio.shape[0]),
                "num_boundaries": len(boundaries),
                "proprio_layout": list(PROPRIO_LAYOUT),
                "action_layout": "next_step_absolute_proprio_proxy",
                "images_imported": bool(extract_images),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return ImportedReassembleEpisode(
        episode_id=episode_id,
        task_id=recording_id,
        recording_id=recording_id,
        num_frames=int(signals.proprio.shape[0]),
        num_boundaries=len(boundaries),
        output_dir=str(episode_dir),
    )


def prompt_records_for_recordings(rows: Iterable[dict[str, Any]]) -> list[PromptRecord]:
    records: list[PromptRecord] = []
    for row in rows:
        recording_id = normalize_text(row.get("recording_id"))
        if not recording_id:
            continue
        high_texts = [normalize_text(text).rstrip(".") for text in row.get("high_level_texts", [])]
        chain = tuple(text for text in high_texts if text and not is_no_action(text))
        task_text = recording_language(row) or recording_id
        records.append(
            PromptRecord(
                task_id=recording_id,
                task_meta_text=task_text,
                primitive_chain=chain[:32],
                prompt=format_prompt(task_text, chain[:32]),
            )
        )
    return records


def write_prompt_artifacts(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    feature_dim: int,
    seed: int,
) -> dict[str, Any]:
    records = prompt_records_for_recordings(rows)
    if not records:
        raise ValueError("No REASSEMBLE prompt records built.")
    output_dir = Path(output_dir)
    table_path = output_dir / "prompt_table.jsonl"
    feature_path = output_dir / "prompt_features.npz"
    manifest_path = output_dir / "prompt_manifest.json"
    features = encode_prompts_mock(records, feature_dim=feature_dim, seed=seed)
    write_prompt_table(records, table_path)
    write_prompt_feature_store(feature_path, records, features)
    manifest = {
        "encoder": "mock",
        "feature_dim": int(feature_dim),
        "num_prompts": len(records),
        "prompt_table": str(table_path),
        "prompt_features": str(feature_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def import_reassemble_subset(
    dataset_root: str | Path,
    recording_index_jsonl: str | Path,
    output: str | Path,
    split: str = "test_split1",
    camera: str = DEFAULT_CAMERA,
    max_recordings: int = 0,
    min_stage_span: int = 10,
    prompt_output: str | Path | None = None,
    prompt_feature_dim: int = 32,
    seed: int = 42,
    extract_images: bool = False,
    jpeg_quality: int = 95,
    overwrite: bool = False,
) -> list[ImportedReassembleEpisode]:
    dataset_root = Path(dataset_root)
    output = Path(output)
    rows_all = read_jsonl(recording_index_jsonl)
    rows = filter_recordings(rows_all, dataset_root=dataset_root, split=split, max_recordings=max_recordings)
    imported: list[ImportedReassembleEpisode] = []
    skipped: list[dict[str, str]] = []
    for row in rows:
        recording_id = normalize_text(row.get("recording_id"))
        try:
            imported.append(
                import_one_recording(
                    dataset_root=dataset_root,
                    output_root=output,
                    recording_row=row,
                    camera=camera,
                    min_stage_span=min_stage_span,
                    extract_images=extract_images,
                    jpeg_quality=jpeg_quality,
                    overwrite=overwrite,
                )
            )
        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
            skipped.append({"recording_id": recording_id, "reason": str(exc)})

    if not imported:
        raise ValueError(f"No REASSEMBLE recordings imported from {dataset_root}; skipped={skipped[:5]}")
    output.mkdir(parents=True, exist_ok=True)
    prompt_manifest = None
    if prompt_output is not None:
        imported_ids = {item.recording_id for item in imported}
        prompt_manifest = write_prompt_artifacts(
            rows=(row for row in rows if normalize_text(row.get("recording_id")) in imported_ids),
            output_dir=prompt_output,
            feature_dim=prompt_feature_dim,
            seed=seed,
        )
    summary = {
        "dataset_root": str(dataset_root),
        "recording_index_jsonl": str(recording_index_jsonl),
        "split": split,
        "camera": camera,
        "num_requested": len(rows),
        "num_imported": len(imported),
        "num_skipped": len(skipped),
        "skipped": skipped,
        "episodes": [asdict(item) for item in imported],
        "prompt_manifest": prompt_manifest,
    }
    (output / "reassemble_import_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return imported


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import REASSEMBLE recordings into WAM episode layout.")
    parser.add_argument("--dataset-root", required=True, help="Root containing REASSEMBLE data/*.h5.")
    parser.add_argument("--recording-index-jsonl", required=True, help="reassemble_recording_index_v1.jsonl.")
    parser.add_argument("--output", required=True, help="Output WAM episode directory.")
    parser.add_argument("--split", default="test_split1")
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--max-recordings", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--min-stage-span", type=int, default=10)
    parser.add_argument("--prompt-output", default=None)
    parser.add_argument("--prompt-feature-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extract-images", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    imported = import_reassemble_subset(
        dataset_root=args.dataset_root,
        recording_index_jsonl=args.recording_index_jsonl,
        output=args.output,
        split=args.split,
        camera=args.camera,
        max_recordings=args.max_recordings,
        min_stage_span=args.min_stage_span,
        prompt_output=args.prompt_output,
        prompt_feature_dim=args.prompt_feature_dim,
        seed=args.seed,
        extract_images=args.extract_images,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
    )
    print(f"imported {len(imported)} REASSEMBLE episodes to {args.output}")


if __name__ == "__main__":
    main()
