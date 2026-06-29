from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch

from mvp0.features import read_feature_store
from mvp0.features import write_feature_store
from mvp0.prepare_windows import read_episode_metas


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract frozen transformer visual features.")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=768)
    parser.add_argument("--mock", action="store_true", help="Write deterministic random features for tests/smoke runs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="cuda/cpu; defaults to cuda when available.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**32)


def extract_mock_features(
    episodes: str | Path,
    output: str | Path,
    feature_dim: int,
    seed: int,
) -> int:
    specs = read_episode_metas(episodes)
    output = Path(output)
    count = 0
    for spec in specs:
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
            output = model(batch)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if output.ndim > 2:
                output = output.flatten(start_dim=2).mean(dim=-1)
            features.append(output.detach().cpu().float())
    return torch.cat(features, dim=0).numpy().astype(np.float16)


def extract_real_features(
    episodes: str | Path,
    output: str | Path,
    model_name: str,
    image_size: int,
    batch_size: int,
    device_name: str | None,
    overwrite: bool,
) -> int:
    try:
        import timm
        from timm.data import create_transform, resolve_model_data_config
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "timm is required for real feature extraction. Install requirements on the 4090 environment."
        ) from exc

    specs = read_episode_metas(episodes)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to(device)
    data_config = resolve_model_data_config(model)
    data_config["input_size"] = (3, image_size, image_size)
    transform = create_transform(**data_config, is_training=False)
    output = Path(output)

    count = 0
    for spec in specs:
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

        episode_dir = Path(episodes) / spec.episode_id
        feature_map = {}
        for camera in spec.cameras:
            image_paths = image_paths_for_camera(episode_dir, camera)
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
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"wrote real features for {count} episodes to {args.output}")


if __name__ == "__main__":
    main()
