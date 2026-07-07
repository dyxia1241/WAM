from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ppwam.prompts import PromptRecord, encode_prompts_mock, format_prompt
from ppwam.prompts import write_prompt_feature_store, write_prompt_table


PRIMARY_CAMERA = "036422060215"
STAGE_VOCAB = ("approach", "grasp", "move", "release")
GENERIC_PRIMITIVE_CHAIN = (
    "approach interaction target",
    "establish grasp or contact",
    "move or manipulate target",
    "release or finish interaction",
)


@dataclass(frozen=True)
class RH20TSceneSelection:
    scene_dir: str
    task_id: str
    task_description: str = ""


@dataclass(frozen=True)
class ImportedRH20TEpisode:
    episode_id: str
    task_id: str
    source_scene_dir: str
    num_frames: int
    num_boundaries: int
    output_dir: str


@dataclass(frozen=True)
class RH20TSignals:
    timestamps: np.ndarray
    source_frame_indices: np.ndarray
    proprio: np.ndarray
    action: np.ndarray
    speed_ema: np.ndarray
    force_mag_ema: np.ndarray
    contact_mask: np.ndarray
    moving_mask: np.ndarray
    intervals: tuple[dict[str, Any], ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import RH20T extracted scenes into WAM episode layout.")
    parser.add_argument("--source-root", required=True, help="Directory containing extracted RH20T_cfg2 scene dirs.")
    parser.add_argument("--output", required=True, help="Output WAM episode directory.")
    parser.add_argument("--scene-list-json", default=None, help="JSON with selected_scenes or scene_list.")
    parser.add_argument("--task-catalog-json", default=None, help="Optional ProcessBench RH20T task catalog JSON.")
    parser.add_argument("--prompt-output", default=None, help="Optional output dir for mock prompt table/features.")
    parser.add_argument("--camera", default=PRIMARY_CAMERA)
    parser.add_argument("--max-scenes", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--min-interval-span", type=int, default=24)
    parser.add_argument("--min-stage-span", type=int, default=10)
    parser.add_argument("--prompt-feature-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--extract-images", action="store_true", help="Extract selected RGB frames from color.mp4.")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return loaded


def load_task_catalog(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    obj = read_json(path)
    tasks = obj.get("tasks", obj)
    if not isinstance(tasks, dict):
        raise ValueError(f"Invalid task catalog: {path}")
    out: dict[str, str] = {}
    for task_id, payload in tasks.items():
        if isinstance(payload, dict):
            text = payload.get("task_description_english") or payload.get("task_description") or ""
        else:
            text = payload
        out[str(task_id)] = " ".join(str(text).split())
    return out


def prompt_text_for_task(task_id: str, task_description: str) -> str:
    description = " ".join(str(task_description).split())
    if description and description != str(task_id):
        return description
    return f"RH20T manipulation task {task_id}."


def _task_id_from_scene(scene_dir: str) -> str:
    return str(scene_dir).split("_user_")[0]


def load_scene_selections(
    source_root: str | Path,
    scene_list_json: str | Path | None = None,
    task_catalog: dict[str, str] | None = None,
    max_scenes: int = 0,
) -> list[RH20TSceneSelection]:
    source_root = Path(source_root)
    task_catalog = task_catalog or {}
    rows: list[dict[str, Any]]
    if scene_list_json is None:
        rows = [{"scene_dir": path.name, "task_id": _task_id_from_scene(path.name)} for path in source_root.iterdir() if path.is_dir()]
    else:
        obj = read_json(scene_list_json)
        raw_rows = obj.get("selected_scenes", obj.get("scene_list", []))
        if not isinstance(raw_rows, list):
            raise ValueError(f"{scene_list_json} must contain selected_scenes or scene_list.")
        rows = [row for row in raw_rows if isinstance(row, dict)]

    selections: list[RH20TSceneSelection] = []
    seen: set[str] = set()
    for row in rows:
        scene_dir = str(row.get("scene_dir", "")).strip()
        if not scene_dir or scene_dir in seen:
            continue
        scene_root = source_root / scene_dir
        if not scene_root.exists():
            continue
        task_id = str(row.get("task_id") or _task_id_from_scene(scene_dir))
        desc = str(row.get("task_description") or row.get("task_meta_description") or task_catalog.get(task_id, ""))
        selections.append(RH20TSceneSelection(scene_dir=scene_dir, task_id=task_id, task_description=" ".join(desc.split())))
        seen.add(scene_dir)
        if max_scenes > 0 and len(selections) >= max_scenes:
            break
    if not selections:
        raise ValueError(f"No RH20T scenes selected from {source_root}.")
    return selections


def _load_object_npy(path: str | Path) -> Any:
    loaded = np.load(path, allow_pickle=True)
    try:
        return loaded.item()
    except ValueError:
        return loaded


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


def hysteresis_mask(mask_hi: np.ndarray, mask_lo: np.ndarray) -> np.ndarray:
    hi = np.asarray(mask_hi, dtype=bool)
    lo = np.asarray(mask_lo, dtype=bool)
    if hi.shape != lo.shape:
        raise ValueError("mask_hi and mask_lo must have the same shape.")
    out = np.zeros_like(hi, dtype=bool)
    state = False
    for idx in range(int(hi.size)):
        if not state and bool(hi[idx]):
            state = True
        elif state and not bool(lo[idx]):
            state = False
        out[idx] = state
    return out


def mask_runs(mask: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    arr = np.asarray(mask, dtype=bool)
    runs: list[tuple[int, int]] = []
    idx = 0
    n = int(arr.size)
    while idx < n:
        if not bool(arr[idx]):
            idx += 1
            continue
        end = idx + 1
        while end < n and bool(arr[end]):
            end += 1
        if end - idx >= int(min_len):
            runs.append((idx, end - 1))
        idx = end
    return runs


def merge_runs(runs: Iterable[tuple[int, int]], merge_gap: int) -> list[tuple[int, int]]:
    ordered = sorted((int(a), int(b)) for a, b in runs)
    if not ordered:
        return []
    merged: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
    for start, end in ordered[1:]:
        current = merged[-1]
        if start - current[1] - 1 <= int(merge_gap):
            current[1] = max(current[1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def build_intervals(
    contact_mask: np.ndarray,
    moving_mask: np.ndarray,
    force_mag: np.ndarray,
    speed: np.ndarray,
    min_interval_span: int,
    merge_gap: int = 20,
) -> tuple[dict[str, Any], ...]:
    contact_runs = mask_runs(contact_mask, min_len=6)
    moving_runs = mask_runs(moving_mask, min_len=8)
    merged = merge_runs([*contact_runs, *moving_runs], merge_gap=merge_gap)
    intervals: list[dict[str, Any]] = []
    for interval_index, (start, end) in enumerate(merged, start=1):
        if end - start + 1 < int(min_interval_span):
            continue
        local_contact = np.where(contact_mask[start : end + 1])[0]
        first_contact = int(start + local_contact[0]) if local_contact.size else None
        last_contact = int(start + local_contact[-1]) if local_contact.size else None
        intervals.append(
            {
                "interval_id": f"i{interval_index:03d}",
                "start": int(start),
                "end": int(end),
                "first_contact": first_contact,
                "last_contact": last_contact,
                "peak_force_row": int(start + np.argmax(force_mag[start : end + 1])),
                "peak_speed_row": int(start + np.argmax(speed[start : end + 1])),
                "contact_ratio": float(np.mean(contact_mask[start : end + 1])),
                "moving_ratio": float(np.mean(moving_mask[start : end + 1])),
            }
        )
    return tuple(intervals)


def _valid_signal_rows(tcp_rows: list[dict[str, Any]], ft_rows: list[dict[str, Any]], timestamps: np.ndarray, gripper: dict[Any, Any]) -> np.ndarray:
    valid = []
    for row, ft_row, ts in zip(tcp_rows, ft_rows, timestamps, strict=False):
        grip = gripper.get(int(ts))
        valid.append(
            isinstance(row, dict)
            and isinstance(ft_row, dict)
            and row.get("tcp") is not None
            and ft_row.get("zeroed") is not None
            and isinstance(grip, dict)
            and grip.get("gripper_info") is not None
        )
    return np.asarray(valid, dtype=bool)


def load_scene_signals(
    scene_root: str | Path,
    camera: str = PRIMARY_CAMERA,
    min_interval_span: int = 24,
) -> RH20TSignals:
    scene_root = Path(scene_root)
    camera_ts_obj = _load_object_npy(scene_root / f"cam_{camera}" / "timestamps.npy")
    camera_ts = np.asarray(camera_ts_obj["color"], dtype=np.int64)
    tcp_rows = list(_load_object_npy(scene_root / "transformed" / "tcp_base.npy")[camera])
    ft_rows = list(_load_object_npy(scene_root / "transformed" / "force_torque_base.npy")[camera])
    gripper = _load_object_npy(scene_root / "transformed" / "gripper.npy")[camera]

    row_ts = np.asarray([int(row["timestamp"]) for row in tcp_rows], dtype=np.int64)
    common = min(int(camera_ts.shape[0]), int(row_ts.shape[0]), len(ft_rows))
    if common <= 0:
        raise ValueError(f"Empty synchronized streams for {scene_root}.")
    camera_ts = camera_ts[:common]
    row_ts = row_ts[:common]
    tcp_rows = tcp_rows[:common]
    ft_rows = ft_rows[:common]
    if not np.array_equal(camera_ts, row_ts):
        if not np.array_equal(camera_ts[:common], row_ts[:common]):
            raise ValueError(f"Timestamp mismatch for {scene_root}.")

    valid = _valid_signal_rows(tcp_rows, ft_rows, row_ts, gripper)
    if not np.any(valid):
        raise ValueError(f"No valid signal rows for {scene_root}.")
    valid_indices = np.where(valid)[0]
    source_frame_indices = valid_indices.astype(np.int64)
    timestamps = row_ts[valid_indices]
    tcp_rows = [tcp_rows[int(idx)] for idx in valid_indices]
    ft_rows = [ft_rows[int(idx)] for idx in valid_indices]
    if len(tcp_rows) < int(min_interval_span):
        raise ValueError(f"Too few valid rows for {scene_root}: {len(tcp_rows)}.")

    tcp = np.stack([np.asarray(row["tcp"][:7], dtype=np.float64) for row in tcp_rows], axis=0)
    force_xyz = np.stack([np.asarray(row["zeroed"][:3], dtype=np.float64) for row in ft_rows], axis=0)
    gripper_width = np.asarray([float(gripper[int(ts)]["gripper_info"][0]) for ts in timestamps], dtype=np.float64)

    dt = np.diff(timestamps, prepend=timestamps[0]).astype(np.float64) / 1000.0
    dt[0] = float(np.median(dt[1:])) if dt.size > 1 else 0.04
    dt = np.clip(dt, 1.0e-3, 1.0)
    pos_delta = np.diff(tcp[:, :3], axis=0, prepend=tcp[[0], :3])
    speed_raw = np.linalg.norm(pos_delta, axis=1) / dt
    speed_ema = ema1d(speed_raw, span=9)
    force_mag_raw = np.linalg.norm(force_xyz, axis=1)
    force_mag_ema = ema1d(force_mag_raw, span=9)

    speed_hi = float(max(0.040, np.percentile(speed_ema, 85)))
    speed_lo = float(max(0.022, speed_hi * 0.55))
    force_hi = float(max(2.50, np.percentile(force_mag_ema, 80)))
    force_lo = float(max(1.50, force_hi * 0.60))
    moving_mask = hysteresis_mask(speed_ema >= speed_hi, speed_ema >= speed_lo)
    contact_mask = hysteresis_mask(force_mag_ema >= force_hi, force_mag_ema >= force_lo)
    intervals = build_intervals(
        contact_mask=contact_mask,
        moving_mask=moving_mask,
        force_mag=force_mag_ema,
        speed=speed_ema,
        min_interval_span=min_interval_span,
    )
    if not intervals:
        raise ValueError(f"No interaction intervals found for {scene_root}.")

    proprio = np.concatenate(
        [
            tcp,
            force_xyz,
            gripper_width[:, None],
            speed_ema[:, None],
        ],
        axis=1,
    ).astype(np.float32)
    action = np.concatenate([proprio[1:], proprio[-1:]], axis=0).astype(np.float32)
    return RH20TSignals(
        timestamps=timestamps,
        source_frame_indices=source_frame_indices,
        proprio=proprio,
        action=action,
        speed_ema=speed_ema.astype(np.float32),
        force_mag_ema=force_mag_ema.astype(np.float32),
        contact_mask=contact_mask,
        moving_mask=moving_mask,
        intervals=intervals,
    )


def interval_to_boundaries(interval: dict[str, Any], min_stage_span: int) -> list[dict[str, Any]]:
    start = int(interval["start"])
    end = int(interval["end"])
    first_contact = interval.get("first_contact")
    last_contact = interval.get("last_contact")
    if first_contact is None:
        return [{"stage": "move", "start": start, "end": end}]

    first_contact = int(first_contact)
    last_contact = int(last_contact) if last_contact is not None else None
    span = end - start + 1
    grasp_end = min(end, first_contact + max(int(min_stage_span) - 1, int(round(span * 0.12))))
    release_start = None
    if last_contact is not None and end - last_contact + 1 >= int(min_stage_span):
        release_start = max(last_contact + 1, end - max(int(min_stage_span) - 1, int(round(span * 0.15))) + 1)

    candidates: list[dict[str, int | str]] = []
    if first_contact - start >= int(min_stage_span):
        candidates.append({"stage": "approach", "start": start, "end": first_contact - 1})
    candidates.append({"stage": "grasp", "start": first_contact, "end": grasp_end})
    move_start = grasp_end + 1
    move_end = (release_start - 1) if release_start is not None else end
    if move_end - move_start + 1 >= int(min_stage_span):
        candidates.append({"stage": "move", "start": move_start, "end": move_end})
    if release_start is not None and end - release_start + 1 >= int(min_stage_span):
        candidates.append({"stage": "release", "start": release_start, "end": end})

    boundaries = [
        {"stage": str(item["stage"]), "start": int(item["start"]), "end": int(item["end"])}
        for item in candidates
        if int(item["end"]) >= int(item["start"]) and int(item["end"]) - int(item["start"]) + 1 >= int(min_stage_span)
    ]
    if not boundaries:
        return [{"stage": "move", "start": start, "end": end}]
    return boundaries


def build_primitive_boundaries(intervals: Iterable[dict[str, Any]], min_stage_span: int = 10) -> list[dict[str, Any]]:
    boundaries: list[dict[str, Any]] = []
    for interval in intervals:
        for boundary in interval_to_boundaries(interval, min_stage_span=min_stage_span):
            if boundaries and int(boundary["start"]) <= int(boundaries[-1]["end"]):
                boundary = dict(boundary)
                boundary["start"] = int(boundaries[-1]["end"]) + 1
            if int(boundary["end"]) >= int(boundary["start"]):
                boundaries.append(boundary)
    if not boundaries:
        raise ValueError("No primitive boundaries built from intervals.")
    return boundaries


def estimate_fps(timestamps: np.ndarray) -> int:
    if timestamps.size < 2:
        return 25
    diffs = np.diff(timestamps.astype(np.float64)) / 1000.0
    median = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 0.04
    if median <= 0:
        return 25
    return max(1, int(round(1.0 / median)))


def extract_video_frames_by_indices(
    video: str | Path,
    output_dir: str | Path,
    frame_indices: np.ndarray,
    jpeg_quality: int,
    overwrite: bool,
) -> int:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("opencv-python-headless is required for --extract-images.") from exc

    video = Path(video)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for image in output_dir.glob("*.jpg"):
            image.unlink()
    existing = sorted(output_dir.glob("*.jpg"))
    if existing and not overwrite:
        if len(existing) != len(frame_indices):
            raise ValueError(f"{output_dir} has {len(existing)} images; expected {len(frame_indices)}.")
        return len(existing)

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video}")
    targets = {int(src): out_idx for out_idx, src in enumerate(frame_indices.tolist())}
    max_target = max(targets) if targets else -1
    read_index = 0
    wrote = 0
    try:
        while read_index <= max_target:
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"{video} ended before source frame {max_target}.")
            if read_index in targets:
                out_idx = targets[read_index]
                out_path = output_dir / f"{out_idx:06d}.jpg"
                if not cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]):
                    raise ValueError(f"Failed to write {out_path}.")
                wrote += 1
            read_index += 1
    finally:
        capture.release()
    return wrote


def safe_episode_id(scene_dir: str) -> str:
    return str(scene_dir).replace("/", "_")


def import_one_scene(
    source_root: Path,
    output_root: Path,
    selection: RH20TSceneSelection,
    camera: str,
    min_interval_span: int,
    min_stage_span: int,
    extract_images: bool,
    jpeg_quality: int,
    overwrite: bool,
) -> ImportedRH20TEpisode:
    scene_root = source_root / selection.scene_dir
    signals = load_scene_signals(scene_root, camera=camera, min_interval_span=min_interval_span)
    boundaries = build_primitive_boundaries(signals.intervals, min_stage_span=min_stage_span)
    episode_id = safe_episode_id(selection.scene_dir)
    episode_dir = output_root / episode_id
    if episode_dir.exists() and overwrite:
        shutil.rmtree(episode_dir)
    if episode_dir.exists() and not overwrite:
        raise FileExistsError(f"{episode_dir} exists. Pass --overwrite to replace it.")
    episode_dir.mkdir(parents=True, exist_ok=True)

    if extract_images:
        extract_video_frames_by_indices(
            video=scene_root / f"cam_{camera}" / "color.mp4",
            output_dir=episode_dir / "images" / camera,
            frame_indices=signals.source_frame_indices,
            jpeg_quality=jpeg_quality,
            overwrite=overwrite,
        )

    metadata = read_json(scene_root / "metadata.json")
    meta = {
        "episode_id": episode_id,
        "task_id": selection.task_id,
        "source": "rh20t",
        "source_scene_dir": selection.scene_dir,
        "language": selection.task_description,
        "fps": estimate_fps(signals.timestamps),
        "num_frames": int(signals.proprio.shape[0]),
        "cameras": [camera],
        "action_dim": int(signals.action.shape[1]),
        "proprio_dim": int(signals.proprio.shape[1]),
        "action_space": "absolute_tcp_force_proxy",
        "images_imported": bool(extract_images),
        "rating": int(metadata.get("rating", -1)),
        "calib_quality": int(metadata.get("calib_quality", -1)),
    }
    (episode_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(
        episode_dir / "arrays.npz",
        proprio=signals.proprio,
        action=signals.action,
        timestamp=signals.timestamps.astype(np.int64),
        frame_index=np.arange(signals.proprio.shape[0], dtype=np.int64),
        source_frame_index=signals.source_frame_indices.astype(np.int64),
        force_mag=signals.force_mag_ema.astype(np.float32),
        tcp_speed=signals.speed_ema.astype(np.float32),
        contact_mask=signals.contact_mask.astype(np.bool_),
        moving_mask=signals.moving_mask.astype(np.bool_),
    )
    labels = {
        "success": bool(int(metadata.get("rating", 0)) >= 4),
        "label_source": "rh20t_force_tcp_interval_v1",
        "source_scene_dir": selection.scene_dir,
        "source_task_id": selection.task_id,
        "params": {
            "camera": camera,
            "min_interval_span": int(min_interval_span),
            "min_stage_span": int(min_stage_span),
            "stage_vocab": list(STAGE_VOCAB),
        },
        "primitive_boundaries": boundaries,
        "primitive_metadata": [
            {
                **interval,
                "source_scene_dir": selection.scene_dir,
            }
            for interval in signals.intervals
        ],
    }
    (episode_dir / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True), encoding="utf-8")
    (episode_dir / "import_manifest.json").write_text(
        json.dumps(
            {
                "source_root": str(source_root),
                "source_scene_dir": selection.scene_dir,
                "camera": camera,
                "num_frames": int(signals.proprio.shape[0]),
                "num_intervals": len(signals.intervals),
                "num_boundaries": len(boundaries),
                "proprio_layout": [
                    "tcp_x",
                    "tcp_y",
                    "tcp_z",
                    "tcp_qx",
                    "tcp_qy",
                    "tcp_qz",
                    "tcp_qw",
                    "force_x",
                    "force_y",
                    "force_z",
                    "gripper_width",
                    "tcp_speed_ema",
                ],
                "action_layout": "next_step_absolute_proprio_proxy",
                "images_imported": bool(extract_images),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return ImportedRH20TEpisode(
        episode_id=episode_id,
        task_id=selection.task_id,
        source_scene_dir=selection.scene_dir,
        num_frames=int(signals.proprio.shape[0]),
        num_boundaries=len(boundaries),
        output_dir=str(episode_dir),
    )


def write_prompt_artifacts(
    selections: Iterable[RH20TSceneSelection],
    output_dir: str | Path,
    feature_dim: int,
    seed: int,
) -> dict[str, Any]:
    by_task: dict[str, str] = {}
    for selection in selections:
        by_task.setdefault(selection.task_id, prompt_text_for_task(selection.task_id, selection.task_description))
    records = [
        PromptRecord(
            task_id=task_id,
            task_meta_text=description,
            primitive_chain=GENERIC_PRIMITIVE_CHAIN,
            prompt=format_prompt(description, GENERIC_PRIMITIVE_CHAIN),
        )
        for task_id, description in sorted(by_task.items())
    ]
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


def import_rh20t_subset(
    source_root: str | Path,
    output: str | Path,
    scene_list_json: str | Path | None = None,
    task_catalog_json: str | Path | None = None,
    prompt_output: str | Path | None = None,
    camera: str = PRIMARY_CAMERA,
    max_scenes: int = 0,
    min_interval_span: int = 24,
    min_stage_span: int = 10,
    prompt_feature_dim: int = 32,
    seed: int = 42,
    extract_images: bool = False,
    jpeg_quality: int = 95,
    overwrite: bool = False,
) -> list[ImportedRH20TEpisode]:
    source_root = Path(source_root)
    output = Path(output)
    task_catalog = load_task_catalog(task_catalog_json)
    selections = load_scene_selections(
        source_root=source_root,
        scene_list_json=scene_list_json,
        task_catalog=task_catalog,
        max_scenes=max_scenes,
    )
    imported: list[ImportedRH20TEpisode] = []
    skipped: list[dict[str, str]] = []
    for selection in selections:
        try:
            imported.append(
                import_one_scene(
                    source_root=source_root,
                    output_root=output,
                    selection=selection,
                    camera=camera,
                    min_interval_span=min_interval_span,
                    min_stage_span=min_stage_span,
                    extract_images=extract_images,
                    jpeg_quality=jpeg_quality,
                    overwrite=overwrite,
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            skipped.append({"scene_dir": selection.scene_dir, "reason": str(exc)})

    if not imported:
        raise ValueError(f"No RH20T scenes imported from {source_root}; skipped={skipped[:5]}")
    output.mkdir(parents=True, exist_ok=True)
    prompt_manifest = None
    if prompt_output is not None:
        prompt_manifest = write_prompt_artifacts(
            selections=(selection for selection in selections if safe_episode_id(selection.scene_dir) in {item.episode_id for item in imported}),
            output_dir=prompt_output,
            feature_dim=prompt_feature_dim,
            seed=seed,
        )
    summary = {
        "source_root": str(source_root),
        "scene_list_json": str(scene_list_json) if scene_list_json is not None else None,
        "camera": camera,
        "num_requested": len(selections),
        "num_imported": len(imported),
        "num_skipped": len(skipped),
        "skipped": skipped,
        "episodes": [asdict(item) for item in imported],
        "prompt_manifest": prompt_manifest,
    }
    (output / "rh20t_import_manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return imported


def main() -> None:
    args = build_parser().parse_args()
    imported = import_rh20t_subset(
        source_root=args.source_root,
        output=args.output,
        scene_list_json=args.scene_list_json,
        task_catalog_json=args.task_catalog_json,
        prompt_output=args.prompt_output,
        camera=args.camera,
        max_scenes=args.max_scenes,
        min_interval_span=args.min_interval_span,
        min_stage_span=args.min_stage_span,
        prompt_feature_dim=args.prompt_feature_dim,
        seed=args.seed,
        extract_images=args.extract_images,
        jpeg_quality=args.jpeg_quality,
        overwrite=args.overwrite,
    )
    print(f"imported {len(imported)} RH20T episodes to {args.output}")


if __name__ == "__main__":
    main()
