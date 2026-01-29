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

echo "CI PASS psu_bundle"
