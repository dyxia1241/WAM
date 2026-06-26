from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MVP-0 plotting entrypoint.")
    parser.add_argument("--eval", default="outputs/eval", help="Evaluation output directory.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("MVP-0 plot stub")
    print(f"eval_dir={args.eval}")


if __name__ == "__main__":
    main()

