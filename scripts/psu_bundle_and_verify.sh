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

# 4) Batch IMU→GRF API JSON output
BATCH_DIR="$OUT_DIR/batch_inputs"
mkdir -p "$BATCH_DIR"
cp tests/fixtures/imu_sample.csv "$BATCH_DIR/a.csv"
cp tests/fixtures/imu_sample.csv "$BATCH_DIR/b.csv"

bash scripts/run_imu_to_grf_batch.sh "$BATCH_DIR" >"$OUT_DIR/imu_to_grf_batch_output.json"

# 5) Batch IMU→GRF perf summary text (extracted from batch JSON)
python3 - "$OUT_DIR/imu_to_grf_batch_output.json" << 'PY' >"$OUT_DIR/imu_to_grf_batch_perf.txt"
from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("expected batch JSON path")

    path = Path(sys.argv[1])
    data = json.loads(path.read_text(encoding="utf-8"))

    summary = dict(data.get("summary", {}))
    meta = dict(data.get("metadata", {}))

    p50 = float(summary.get("p50_total_ms", 0.0))
    p95 = float(summary.get("p95_total_ms", 0.0))
    max_rss = float(summary.get("max_rss_mb", 0.0))
    num_files = int(meta.get("num_files", 0))
    num_ok = int(meta.get("num_ok", 0))
    num_failed = int(meta.get("num_failed", 0))

    print(f"P50_total_ms={p50}")
    print(f"P95_total_ms={p95}")
    print(f"MAX_rss_mb={max_rss}")
    print(f"num_files={num_files}")
    print(f"num_ok={num_ok}")
    print(f"num_failed={num_failed}")


if __name__ == "__main__":
    main()
PY

# 6) Bundle manifest listing all files in OUT_DIR
bash scripts/write_bundle_manifest.sh "$OUT_DIR"

echo "== PSU_BUNDLE done =="
echo "OUT_DIR=$OUT_DIR"
echo "FILES:"
echo "- imu_to_grf_output.json"
echo "- imu_to_grf_perf.txt"
echo "- provenance.txt"
echo "- bundle_manifest.txt"
echo "- imu_to_grf_batch_output.json"
echo "- imu_to_grf_batch_perf.txt"
