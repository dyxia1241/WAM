from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract frozen transformer visual features.")
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def main() -> None:
    build_parser().parse_args()
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

