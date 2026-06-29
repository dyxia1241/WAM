from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mvp0.features import read_feature_store
from mvp0.features import write_feature_store
from mvp0.prepare_windows import read_episode_meta
from mvp0.schemas import EpisodeMeta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract frozen transformer visual features.")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--raw-gm100-root", default=None, help="Read GM-100 videos directly when episode images are absent.")
    parser.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=768)
    parser.add_argument("--mock", action="store_true", help="Write deterministic random features for tests/smoke runs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-missing-labels", action="store_true")
    parser.add_argument("--limit-episodes", type=int, default=0, help="0 means no limit.")
    parser.add_argument("--device", default=None, help="cuda/cpu; defaults to cuda when available.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return loaded


def episode_records(
    episodes: str | Path,
    skip_missing_labels: bool = False,
    limit_episodes: int = 0,
) -> list[tuple[Path, EpisodeMeta, dict[str, Any]]]:
    episodes = Path(episodes)
    if not episodes.exists():
        raise FileNotFoundError(episodes)
    records: list[tuple[Path, EpisodeMeta, dict[str, Any]]] = []
    for episode_dir in sorted(path for path in episodes.iterdir() if path.is_dir()):
        if skip_missing_labels and not (episode_dir / "labels.json").exists():
            continue
        meta = read_episode_meta(episode_dir, validate_arrays=False)
        meta_json = read_json(episode_dir / "meta.json")
        records.append((episode_dir, meta, meta_json))
        if limit_episodes > 0 and len(records) >= limit_episodes:
            break
    if not records:
        raise ValueError(f"No episode directories selected from {episodes}.")
    return records


def extract_mock_features(
    episodes: str | Path,
    output: str | Path,
    feature_dim: int,
    seed: int,
    skip_missing_labels: bool = False,
    limit_episodes: int = 0,
) -> int:
    records = episode_records(episodes, skip_missing_labels=skip_missing_labels, limit_episodes=limit_episodes)
    output = Path(output)
    count = 0
    for _, spec, _ in records:
        features = {}
        for camera in spec.cameras:
            rng = np.random.default_rng(stable_seed(seed, spec.episode_id, camera))
            features[camera] = rng.normal(
                size=(spec.num_frames, feature_dim),
            ).astype(np.float16)
        write_feature_store(output / f"{spec.episode_id}.npz", features)
        count += 1
    return count


def image_paths_for_camera(episode_dir: str | Path, camera: str) -> list[Path]:
    camera_dir = Path(episode_dir) / "images" / camera
    if not camera_dir.exists():
        raise FileNotFoundError(camera_dir)
    paths = sorted(
        path
        for path in camera_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not paths:
        raise ValueError(f"No image files found in {camera_dir}.")
    return paths


def _load_image(path: Path):
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for real feature extraction.") from exc
    with Image.open(path) as image:
        return image.convert("RGB")


def _image_from_bgr_frame(frame: np.ndarray):
    try:
        from PIL import Image
    except ModuleNotFoundError as exc:
        raise RuntimeError("Pillow is required for real feature extraction.") from exc
    return Image.fromarray(frame[:, :, ::-1]).convert("RGB")


def encode_tensor_batch(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    output = model(batch)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim > 2:
        output = output.flatten(start_dim=2).mean(dim=-1)
    return output.detach().cpu().float()


def extract_camera_features(
    model: torch.nn.Module,
    transform,
    image_paths: list[Path],
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    features: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start : start + batch_size]
            batch = torch.stack([transform(_load_image(path)) for path in batch_paths], dim=0).to(device)
            features.append(encode_tensor_batch(model, batch))
    return torch.cat(features, dim=0).numpy().astype(np.float16)


def gm100_video_path(raw_gm100_root: str | Path, task_id: str, source_episode_id: str, camera: str) -> Path:
    return (
        Path(raw_gm100_root)
        / task_id
        / "videos"
        / "chunk-000"
        / f"observation.images.{camera}"
        / f"{source_episode_id}.mp4"
    )


def extract_video_camera_features(
    model: torch.nn.Module,
    transform,
    video_path: str | Path,
    expected_frames: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError("opencv-python-headless is required to read GM-100 videos.") from exc

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    features: list[torch.Tensor] = []
    batch_images: list[torch.Tensor] = []
    frames_read = 0
    model.eval()
    try:
        with torch.no_grad():
            while frames_read < expected_frames:
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"{video_path} ended at frame {frames_read}; expected {expected_frames}.")
                batch_images.append(transform(_image_from_bgr_frame(frame)))
                frames_read += 1
                if len(batch_images) == batch_size:
                    batch = torch.stack(batch_images, dim=0).to(device)
                    features.append(encode_tensor_batch(model, batch))
                    batch_images.clear()
            if batch_images:
                batch = torch.stack(batch_images, dim=0).to(device)
                features.append(encode_tensor_batch(model, batch))
    finally:
        capture.release()
    return torch.cat(features, dim=0).numpy().astype(np.float16)


def extract_real_features(
    episodes: str | Path,
    output: str | Path,
    model_name: str,
    image_size: int,
    batch_size: int,
    device_name: str | None,
    overwrite: bool,
    raw_gm100_root: str | Path | None = None,
    skip_missing_labels: bool = False,
    limit_episodes: int = 0,
) -> int:
    try:
        import timm
        from timm.data import create_transform, resolve_model_data_config
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "timm is required for real feature extraction. Install requirements on the 4090 environment."
        ) from exc

    records = episode_records(episodes, skip_missing_labels=skip_missing_labels, limit_episodes=limit_episodes)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device)
    data_config = resolve_model_data_config(model)
    data_config["input_size"] = (3, image_size, image_size)
    transform = create_transform(**data_config, is_training=False)
    output = Path(output)

    count = 0
    for episode_dir, spec, meta_json in records:
        output_path = output / f"{spec.episode_id}.npz"
        if output_path.exists() and not overwrite:
            try:
                read_feature_store(
                    output_path,
                    expected_cameras=spec.cameras,
                    expected_frames=spec.num_frames,
                )
                continue
            except ValueError:
                pass

        feature_map = {}
        for camera in spec.cameras:
            try:
                image_paths = image_paths_for_camera(episode_dir, camera)
            except FileNotFoundError:
                if raw_gm100_root is None:
                    raise
                source_episode_id = str(meta_json.get("source_episode_id", ""))
                if not source_episode_id:
                    raise ValueError(f"{spec.episode_id} meta.json must contain source_episode_id for raw video extraction.")
                feature_map[camera] = extract_video_camera_features(
                    model=model,
                    transform=transform,
                    video_path=gm100_video_path(raw_gm100_root, spec.task_id, source_episode_id, camera),
                    expected_frames=spec.num_frames,
                    batch_size=batch_size,
                    device=device,
                )
            else:
                if len(image_paths) != spec.num_frames:
                    raise ValueError(
                        f"{spec.episode_id}/{camera} has {len(image_paths)} images; "
                        f"expected {spec.num_frames}."
                    )
                feature_map[camera] = extract_camera_features(
                    model=model,
                    transform=transform,
                    image_paths=image_paths,
                    batch_size=batch_size,
                    device=device,
                )
        write_feature_store(output_path, feature_map)
        count += 1
    return count


def main() -> None:
    args = build_parser().parse_args()
    if args.mock:
        count = extract_mock_features(
            episodes=args.episodes,
            output=args.output,
            feature_dim=args.feature_dim,
            seed=args.seed,
            skip_missing_labels=args.skip_missing_labels,
            limit_episodes=args.limit_episodes,
        )
        print(f"wrote mock features for {count} episodes to {args.output}")
        return

    try:
        count = extract_real_features(
            episodes=args.episodes,
            output=args.output,
            model_name=args.model,
            image_size=args.image_size,
            batch_size=args.batch_size,
            device_name=args.device,
            overwrite=args.overwrite,
            raw_gm100_root=args.raw_gm100_root,
            skip_missing_labels=args.skip_missing_labels,
            limit_episodes=args.limit_episodes,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"wrote real features for {count} episodes to {args.output}")


if __name__ == "__main__":
    main()
