from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MVP-0 training entrypoint.")
    parser.add_argument("--config", default="mvp0/configs/debug.yaml")
    parser.add_argument("overrides", nargs="*", help="Optional key=value overrides.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("MVP-0 train stub")
    print(f"config={args.config}")
    if args.overrides:
        print("overrides=" + " ".join(args.overrides))
    print("Full training loop is intentionally deferred until toy tests pass.")


if __name__ == "__main__":
    main()

