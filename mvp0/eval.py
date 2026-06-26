from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MVP-0 evaluation entrypoint.")
    parser.add_argument("--checkpoint", required=False)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", default="outputs/eval")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("MVP-0 eval stub")
    print(f"checkpoint={args.checkpoint}")
    print(f"split={args.split}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()

