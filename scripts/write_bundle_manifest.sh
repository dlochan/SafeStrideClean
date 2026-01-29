#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 OUT_DIR" >&2
  exit 1
fi

OUT_DIR="$1"

if [ ! -d "$OUT_DIR" ]; then
  echo "write_bundle_manifest: OUT_DIR is not a directory: $OUT_DIR" >&2
  exit 1
fi

cd "$OUT_DIR"

# List all regular files in OUT_DIR (non-recursive), relative names only, sorted.
{
  for f in *; do
    if [ -f "$f" ]; then
      echo "$f"
    fi
  done
} | sort > bundle_manifest.txt
