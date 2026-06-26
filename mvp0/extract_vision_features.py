from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from mvp0.features import write_feature_store
from mvp0.prepare_windows import read_episode_specs


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
    specs = read_episode_specs(episodes)
    output = Path(output)
    count = 0
    for spec in specs:
        features = {}
        for camera in spec.meta.cameras:
            rng = np.random.default_rng(stable_seed(seed, spec.meta.episode_id, camera))
            features[camera] = rng.normal(
                size=(spec.meta.num_frames, feature_dim),
            ).astype(np.float16)
        write_feature_store(output / f"{spec.meta.episode_id}.npz", features)
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
        import timm  # noqa: F401
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "timm is required for real feature extraction. "
            "Run this entrypoint on the 4090 environment after installing requirements."
        ) from exc
    raise SystemExit("Real image feature extraction is intentionally deferred to the 4090 workflow.")


if __name__ == "__main__":
    main()
