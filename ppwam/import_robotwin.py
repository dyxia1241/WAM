from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


DEFAULT_STAGE_SEQUENCE = ("approach", "grasp", "move", "release")
ROBOTWIN_CAMERA_SUFFIX = "_camera"


@dataclass(frozen=True)
class ImportedRoboTwinEpisode:
    episode_id: str
    task_id: str
    source_hdf5: str
    num_frames: int
    cameras: tuple[str, ...]
    output_dir: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import RoboTwin HDF5 episodes into WAM episode layout.")
    parser.add_argument("--hdf5", nargs="+", required=True, help="One or more RoboTwin episode*.hdf5 files.")
    parser.add_argument("--output", required=True, help="Output WAM episode directory.")
    parser.add_argument("--task-id", default="", help="Defaults to the task directory inferred from RoboTwin data layout.")
    parser.add_argument("--language", default="", help="Optional language instruction override.")
    parser.add_argument("--instructions", default="", help="Optional RoboTwin instructions/episodeN.json path.")
    parser.add_argument("--scene-info", default="", help="Optional RoboTwin scene_info.json path.")
    parser.add_argument("--label-sidecar", default="", help="Optional JSON with primitive_boundaries and potential/phi labels.")
    parser.add_argument("--episode-id-prefix", default="", help="Optional prefix for generated WAM episode ids.")
    parser.add_argument("--stage-sequence", default=",".join(DEFAULT_STAGE_SEQUENCE))
    parser.add_argument("--max-frames", type=int, default=None, help="Optional small smoke-test frame cap.")
    parser.add_argument("--skip-images", action="store_true", help="Only write metadata and arrays; do not extract RGB JPEGs.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return loaded


def parse_stage_sequence(text: str) -> tuple[str, ...]:
    stages = tuple(item.strip() for item in text.split(",") if item.strip())
    if not stages:
        raise ValueError("stage sequence must not be empty.")
    allowed = {"approach", "grasp", "move", "place", "release"}
    unknown = [stage for stage in stages if stage not in allowed]
    if unknown:
        raise ValueError(f"Unknown RoboTwin stage names: {unknown}")
    return stages


def infer_task_id(hdf5_path: str | Path) -> str:
    path = Path(hdf5_path)
    try:
        if path.parent.name == "data" and path.parent.parent.parent.name:
            return path.parent.parent.parent.name
    except IndexError:
        pass
    return path.parent.parent.name if path.parent.parent.name else "robotwin_task"


def episode_number(path: str | Path) -> int | None:
    match = re.search(r"episode(\d+)", Path(path).stem)
    return int(match.group(1)) if match else None


def default_episode_id(hdf5_path: str | Path, task_id: str, prefix: str = "") -> str:
    base = Path(hdf5_path).stem
    stem = f"{task_id}_{base}"
    return f"{prefix}_{stem}" if prefix else stem


def source_config_name(hdf5_path: str | Path) -> str:
    path = Path(hdf5_path)
    if path.parent.name == "data":
        return path.parent.parent.name
    return ""


def camera_names_from_hdf5(handle: h5py.File) -> tuple[str, ...]:
    observation = handle.get("observation")
    if observation is None:
        return tuple()
    names = [
        str(name)
        for name, group in observation.items()
        if isinstance(group, h5py.Group) and "rgb" in group
    ]
    return tuple(sorted(names))


def read_dataset(handle: h5py.File, name: str, dtype: np.dtype | type = np.float32) -> np.ndarray:
    if name not in handle:
        raise ValueError(f"Missing RoboTwin HDF5 dataset: {name}")
    array = np.asarray(handle[name][()], dtype=dtype)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"Dataset {name} must be 2D after loading; got shape {array.shape}.")
    return array


def read_vector_dataset(handle: h5py.File, name: str, dtype: np.dtype | type = np.float32) -> np.ndarray:
    if name not in handle:
        raise ValueError(f"Missing RoboTwin HDF5 dataset: {name}")
    array = np.asarray(handle[name][()], dtype=dtype)
    if array.ndim != 1:
        raise ValueError(f"Dataset {name} must be 1D; got shape {array.shape}.")
    return array


def read_robotwin_arrays(hdf5_path: str | Path) -> dict[str, np.ndarray]:
    with h5py.File(hdf5_path, "r") as handle:
        joint_action = read_dataset(handle, "joint_action/vector")
        left_endpose = read_dataset(handle, "endpose/left_endpose")
        right_endpose = read_dataset(handle, "endpose/right_endpose")
        left_gripper = read_vector_dataset(handle, "endpose/left_gripper")[:, None]
        right_gripper = read_vector_dataset(handle, "endpose/right_gripper")[:, None]
        proprio = np.concatenate(
            [joint_action, left_endpose, left_gripper, right_endpose, right_gripper],
            axis=1,
        ).astype(np.float32)
    action = np.concatenate([proprio[1:], proprio[-1:]], axis=0).astype(np.float32)
    num_frames = int(proprio.shape[0])
    return {
        "proprio": proprio,
        "action": action,
        "joint_action_vector": joint_action.astype(np.float32),
        "left_endpose": left_endpose.astype(np.float32),
        "right_endpose": right_endpose.astype(np.float32),
        "left_gripper": left_gripper.squeeze(axis=1).astype(np.float32),
        "right_gripper": right_gripper.squeeze(axis=1).astype(np.float32),
        "frame_index": np.arange(num_frames, dtype=np.int64),
    }


def slice_arrays(arrays: dict[str, np.ndarray], num_frames: int) -> dict[str, np.ndarray]:
    return {key: value[:num_frames].copy() for key, value in arrays.items()}


def jpeg_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        raw = value
    else:
        raw = bytes(value)
    return raw.rstrip(b"\0")


def clear_image_dir(path: Path) -> None:
    if not path.exists():
        return
    for image in path.iterdir():
        if image.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            image.unlink()


def extract_hdf5_jpegs(
    hdf5_path: str | Path,
    output_dir: str | Path,
    cameras: Iterable[str],
    num_frames: int,
    overwrite: bool,
) -> int:
    output_dir = Path(output_dir)
    wrote = 0
    with h5py.File(hdf5_path, "r") as handle:
        for camera in cameras:
            rgb_path = f"observation/{camera}/rgb"
            if rgb_path not in handle:
                raise ValueError(f"Missing RoboTwin HDF5 dataset: {rgb_path}")
            camera_dir = output_dir / camera
            existing = sorted(camera_dir.glob("*.jpg"))
            if existing and not overwrite:
                if len(existing) != int(num_frames):
                    raise ValueError(f"{camera_dir} has {len(existing)} images; expected {num_frames}.")
                wrote += len(existing)
                continue
            camera_dir.mkdir(parents=True, exist_ok=True)
            if overwrite:
                clear_image_dir(camera_dir)
            rgb = handle[rgb_path]
            for index in range(int(num_frames)):
                (camera_dir / f"{index:06d}.jpg").write_bytes(jpeg_bytes(rgb[index]))
                wrote += 1
    return wrote


def even_primitive_boundaries(num_frames: int, stages: Iterable[str]) -> list[dict[str, Any]]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    stage_list = list(stages)
    if not stage_list:
        raise ValueError("stages must not be empty.")
    stage_list = stage_list[:num_frames]
    base = int(num_frames) // len(stage_list)
    remainder = int(num_frames) % len(stage_list)
    boundaries: list[dict[str, Any]] = []
    start = 0
    for index, stage in enumerate(stage_list):
        span = base + (1 if index < remainder else 0)
        end = start + span - 1
        boundaries.append({"stage": stage, "start": int(start), "end": int(end)})
        start = end + 1
    return boundaries


def linear_potential(num_frames: int) -> list[float]:
    if num_frames <= 1:
        return [1.0] * max(1, int(num_frames))
    return [float(item) for item in np.linspace(0.0, 1.0, int(num_frames), dtype=np.float32)]


def read_instruction_text(path: str | Path, episode_idx: int | None = None) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    obj = read_json(p)
    if "seen" in obj and isinstance(obj["seen"], list) and obj["seen"]:
        return str(obj["seen"][0])
    if "unseen" in obj and isinstance(obj["unseen"], list) and obj["unseen"]:
        return str(obj["unseen"][0])
    if episode_idx is not None:
        payload = obj.get(f"episode_{episode_idx}")
        if isinstance(payload, dict):
            for key in ("instruction", "language", "task_description"):
                if payload.get(key):
                    return str(payload[key])
    for key in ("instruction", "language", "task_description"):
        if obj.get(key):
            return str(obj[key])
    return ""


def read_scene_episode(path: str | Path, episode_idx: int | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    obj = read_json(p)
    if episode_idx is not None:
        payload = obj.get(f"episode_{episode_idx}")
        return payload if isinstance(payload, dict) else None
    return obj


def select_label_sidecar(path: str | Path, hdf5_path: str | Path, episode_idx: int | None) -> dict[str, Any] | None:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    obj = read_json(p)
    episodes = obj.get("episodes")
    if isinstance(episodes, dict):
        keys = [Path(hdf5_path).stem]
        if episode_idx is not None:
            keys.extend([f"episode{episode_idx}", str(episode_idx), f"episode_{episode_idx}"])
        for key in keys:
            payload = episodes.get(key)
            if isinstance(payload, dict):
                return payload
        raise ValueError(f"No label sidecar entry for {hdf5_path} in {p}.")
    return obj


def validate_boundaries_payload(boundaries: Any, num_frames: int) -> list[dict[str, Any]]:
    if not isinstance(boundaries, list) or not boundaries:
        raise ValueError("label sidecar primitive_boundaries must be a non-empty list.")
    allowed = {"approach", "grasp", "move", "place", "release"}
    out: list[dict[str, Any]] = []
    previous_end = -1
    for item in boundaries:
        if not isinstance(item, dict):
            raise ValueError("primitive_boundaries entries must be objects.")
        stage = str(item["stage"])
        start = int(item["start"])
        end = int(item["end"])
        if stage not in allowed:
            raise ValueError(f"Unknown sidecar stage: {stage}")
        if start < 0 or end < start or end >= int(num_frames):
            raise ValueError(f"Invalid sidecar boundary range: {item}")
        if start <= previous_end:
            raise ValueError("sidecar primitive_boundaries must be sorted and non-overlapping.")
        out.append({"stage": stage, "start": start, "end": end})
        previous_end = end
    return out


def potential_from_sidecar(sidecar: dict[str, Any], num_frames: int) -> list[float] | None:
    raw = sidecar.get("potential", sidecar.get("phi"))
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("label sidecar potential/phi must be a list.")
    if len(raw) < int(num_frames):
        raise ValueError(f"label sidecar potential length={len(raw)} is shorter than num_frames={num_frames}.")
    values = [float(item) for item in raw[: int(num_frames)]]
    if not np.all(np.isfinite(np.asarray(values, dtype=np.float32))):
        raise ValueError("label sidecar potential contains non-finite values.")
    return values


def import_one_hdf5(
    hdf5_path: str | Path,
    output_root: str | Path,
    task_id: str = "",
    language: str = "",
    instructions: str | Path = "",
    scene_info: str | Path = "",
    label_sidecar: str | Path = "",
    episode_id_prefix: str = "",
    stages: Iterable[str] = DEFAULT_STAGE_SEQUENCE,
    max_frames: int | None = None,
    skip_images: bool = False,
    overwrite: bool = False,
) -> ImportedRoboTwinEpisode:
    hdf5_path = Path(hdf5_path)
    output_root = Path(output_root)
    if not hdf5_path.exists():
        raise FileNotFoundError(hdf5_path)
    task_id = task_id or infer_task_id(hdf5_path)
    episode_idx = episode_number(hdf5_path)
    episode_id = default_episode_id(hdf5_path, task_id=task_id, prefix=episode_id_prefix)
    episode_dir = output_root / episode_id
    if episode_dir.exists() and overwrite:
        shutil.rmtree(episode_dir)
    if episode_dir.exists() and not overwrite:
        raise FileExistsError(f"{episode_dir} exists. Pass --overwrite to replace it.")
    episode_dir.mkdir(parents=True, exist_ok=True)

    arrays = read_robotwin_arrays(hdf5_path)
    source_frames = int(arrays["proprio"].shape[0])
    num_frames = source_frames if max_frames is None else min(source_frames, int(max_frames))
    arrays = slice_arrays(arrays, num_frames)
    with h5py.File(hdf5_path, "r") as handle:
        cameras = camera_names_from_hdf5(handle)
    if not cameras:
        raise ValueError(f"No RGB cameras found in {hdf5_path}.")

    images_imported = False
    if not skip_images:
        extract_hdf5_jpegs(
            hdf5_path=hdf5_path,
            output_dir=episode_dir / "images",
            cameras=cameras,
            num_frames=num_frames,
            overwrite=overwrite,
        )
        images_imported = True

    instruction_text = language or read_instruction_text(instructions, episode_idx=episode_idx)
    sidecar = select_label_sidecar(label_sidecar, hdf5_path=hdf5_path, episode_idx=episode_idx)
    boundaries = (
        validate_boundaries_payload(sidecar["primitive_boundaries"], num_frames)
        if sidecar is not None and sidecar.get("primitive_boundaries") is not None
        else even_primitive_boundaries(num_frames, stages=stages)
    )
    sidecar_potential = potential_from_sidecar(sidecar, num_frames) if sidecar is not None else None
    potential = sidecar_potential if sidecar_potential is not None else linear_potential(num_frames)
    label_source = (
        str(sidecar.get("label_source", "robotwin_sidecar_v0"))
        if sidecar is not None
        else "robotwin_linear_progress_v0"
    )
    scene_payload = read_scene_episode(scene_info, episode_idx=episode_idx)
    success = bool(sidecar.get("success", True)) if sidecar is not None else True

    meta = {
        "episode_id": episode_id,
        "task_id": task_id,
        "source": "robotwin",
        "source_hdf5": str(hdf5_path),
        "source_config": source_config_name(hdf5_path),
        "language": instruction_text,
        "fps": 30,
        "num_frames": num_frames,
        "cameras": list(cameras),
        "action_dim": int(arrays["action"].shape[1]),
        "proprio_dim": int(arrays["proprio"].shape[1]),
        "action_space": "next_step_absolute_proprio_proxy",
        "images_imported": images_imported,
        "success": success,
    }
    labels = {
        "success": success,
        "label_source": label_source,
        "source_hdf5": str(hdf5_path),
        "source_episode_index": episode_idx,
        "source_scene_info": scene_payload,
        "params": {
            "stage_sequence": list(stages),
            "potential": "sidecar" if sidecar_potential is not None else "linear_0_1",
            "action": "next_step_absolute_proprio_proxy",
            "label_sidecar": str(label_sidecar) if label_sidecar else None,
        },
        "primitive_boundaries": boundaries,
        "potential": potential,
    }
    if sidecar is not None:
        for key in ("suboptimal_type", "variant_id", "primitive_metadata"):
            if key in sidecar:
                labels[key] = sidecar[key]
    manifest = {
        "source_hdf5": str(hdf5_path),
        "source_frames": source_frames,
        "num_frames": num_frames,
        "camera_names": list(cameras),
        "images_imported": images_imported,
        "proprio_layout": [
            "joint_action_vector[14]",
            "left_endpose[7]",
            "left_gripper[1]",
            "right_endpose[7]",
            "right_gripper[1]",
        ],
        "action_layout": "next_step_absolute_proprio_proxy",
        "label_sidecar": str(label_sidecar) if label_sidecar else None,
        "arrays": sorted(arrays),
    }
    (episode_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    (episode_dir / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True), encoding="utf-8")
    (episode_dir / "import_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    np.savez_compressed(episode_dir / "arrays.npz", **arrays)
    return ImportedRoboTwinEpisode(
        episode_id=episode_id,
        task_id=task_id,
        source_hdf5=str(hdf5_path),
        num_frames=num_frames,
        cameras=cameras,
        output_dir=str(episode_dir),
    )


def import_robotwin_hdf5(
    hdf5_paths: Iterable[str | Path],
    output: str | Path,
    task_id: str = "",
    language: str = "",
    instructions: str | Path = "",
    scene_info: str | Path = "",
    label_sidecar: str | Path = "",
    episode_id_prefix: str = "",
    stage_sequence: str | Iterable[str] = DEFAULT_STAGE_SEQUENCE,
    max_frames: int | None = None,
    skip_images: bool = False,
    overwrite: bool = False,
) -> list[ImportedRoboTwinEpisode]:
    stages = parse_stage_sequence(stage_sequence) if isinstance(stage_sequence, str) else tuple(stage_sequence)
    imported = [
        import_one_hdf5(
            hdf5_path=path,
            output_root=output,
            task_id=task_id,
            language=language,
            instructions=instructions,
            scene_info=scene_info,
            label_sidecar=label_sidecar,
            episode_id_prefix=episode_id_prefix,
            stages=stages,
            max_frames=max_frames,
            skip_images=skip_images,
            overwrite=overwrite,
        )
        for path in hdf5_paths
    ]
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    summary = {
        "num_imported": len(imported),
        "episodes": [asdict(item) for item in imported],
    }
    (output_path / "robotwin_import_manifest.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return imported


def main() -> None:
    args = build_parser().parse_args()
    imported = import_robotwin_hdf5(
        hdf5_paths=args.hdf5,
        output=args.output,
        task_id=args.task_id,
        language=args.language,
        instructions=args.instructions,
        scene_info=args.scene_info,
        label_sidecar=args.label_sidecar,
        episode_id_prefix=args.episode_id_prefix,
        stage_sequence=args.stage_sequence,
        max_frames=args.max_frames,
        skip_images=args.skip_images,
        overwrite=args.overwrite,
    )
    print(f"imported {len(imported)} RoboTwin episodes to {args.output}")


if __name__ == "__main__":
    main()
