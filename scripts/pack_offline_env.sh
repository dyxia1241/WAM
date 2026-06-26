#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <wheelhouse_dir>" >&2
  exit 2
fi

wheelhouse="$1"
mkdir -p "$wheelhouse"

python -m pip download \
  --dest "$wheelhouse" \
  -r requirements.txt

echo "Downloaded wheels to $wheelhouse"

