#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

OUT_DIR="artifacts/psu_demo_bundle_$(date -u +%Y%m%d-%H%M%S)_$(git rev-parse --short HEAD)"
mkdir -p "$OUT_DIR"

echo "== PSU_BUNDLE start =="
echo "OUT_DIR=$OUT_DIR"

# 1) IMU→GRF API JSON output
bash scripts/run_imu_to_grf.sh tests/fixtures/imu_sample.csv >"$OUT_DIR/imu_to_grf_output.json"

# 2) IMU→GRF perf summary text
bash scripts/bench_imu_to_grf.sh >"$OUT_DIR/imu_to_grf_perf.txt"

# 3) Provenance information
bash scripts/write_provenance.sh >"$OUT_DIR/provenance.txt"

# 4) Bundle manifest with SHA-256 per file
(
  cd "$OUT_DIR"
  : >bundle_manifest.txt
  for f in imu_to_grf_output.json imu_to_grf_perf.txt provenance.txt; do
    if [ -f "$f" ]; then
      shasum -a 256 "$f" >>bundle_manifest.txt
    fi
  done
)

echo "== PSU_BUNDLE done =="
echo "OUT_DIR=$OUT_DIR"
echo "FILES:"
echo "- imu_to_grf_output.json"
echo "- imu_to_grf_perf.txt"
echo "- provenance.txt"
echo "- bundle_manifest.txt"
