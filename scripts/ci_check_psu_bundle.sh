#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

OUT="$(bash scripts/psu_bundle_and_verify.sh)"

echo "$OUT"

OUT_DIR="$(printf '%s
' "$OUT" | awk -F= '/^OUT_DIR=/{print $2}' | tail -n 1)"

if [ -z "$OUT_DIR" ] || [ ! -d "$OUT_DIR" ]; then
  echo "CI FAIL psu_bundle: OUT_DIR not found" >&2
  exit 1
fi

expect_files=(
  "imu_to_grf_output.json"
  "imu_to_grf_perf.txt"
  "provenance.txt"
  "bundle_manifest.txt"
  "imu_to_grf_batch_output.json"
  "imu_to_grf_batch_perf.txt"
)

for f in "${expect_files[@]}"; do
  path="$OUT_DIR/$f"
  if [ ! -s "$path" ]; then
    echo "CI FAIL psu_bundle: missing or empty $path" >&2
    exit 1
  fi
done

if ! grep -q '"schema_version": "imu_grf_v1"' "$OUT_DIR/imu_to_grf_output.json"; then
  echo "CI FAIL psu_bundle: imu_to_grf_output.json missing expected schema_version" >&2
  exit 1
fi

OUT_DIR="$OUT_DIR" python3 - << 'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def fail(msg: str) -> None:
    print(f"CI FAIL psu_bundle: {msg}", file=sys.stderr)
    raise SystemExit(1)


out_dir = Path(os.environ.get("OUT_DIR", ""))
if not out_dir.is_dir():
    fail("OUT_DIR invalid in Python validation")

batch_path = out_dir / "imu_to_grf_batch_output.json"
try:
    batch = json.loads(batch_path.read_text(encoding="utf-8"))
except Exception as e:  # pragma: no cover - defensive
    fail(f"failed to load batch JSON: {e}")

if batch.get("schema_version") != "imu_grf_batch_v1":
    fail(f"unexpected batch schema_version={batch.get('schema_version')!r}")

meta = dict(batch.get("metadata", {}))
num_files = int(meta.get("num_files", 0))
num_ok = int(meta.get("num_ok", 0))
num_failed = int(meta.get("num_failed", 0))

if not (num_files == 2 and num_ok == 2 and num_failed == 0):
    fail(
        f"batch counts invalid num_files={num_files} num_ok={num_ok} num_failed={num_failed}"
    )

results = batch.get("results")
if not isinstance(results, list) or not results:
    fail("batch results missing or empty")

first = results[0]
output = first.get("output")
if not isinstance(output, dict):
    fail("first batch result missing output dict")

shape = output.get("shape")
if list(map(int, shape or [])) != [64, 256, 1]:
    fail(f"unexpected first batch output.shape={shape!r}")

PY

echo "CI PASS psu_bundle"
