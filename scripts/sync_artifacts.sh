#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 <source_data_root> <destination>" >&2
  exit 2
fi

source_root="$1"
destination="$2"

rsync -avh --progress \
  --exclude 'raw/' \
  --exclude 'outputs/*/best.pt' \
  "${source_root%/}/" \
  "$destination"

