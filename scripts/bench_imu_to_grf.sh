#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

python3 - << 'PY'
from __future__ import annotations

import json
from pathlib import Path

from src.api import run_imu_to_grf


def _run_once() -> dict:
    out = run_imu_to_grf(
        "tests/fixtures/imu_sample.csv",
        window_len=256,
        stride=1,
        num_windows=64,
        profile=True,
    )
    perf = dict(out.get("perf", {}))
    return {
        "total_ms": float(perf.get("total_ms", 0.0)),
        "forward_ms": float(perf.get("forward_ms", 0.0)),
        "rss_mb": float(perf.get("rss_mb", 0.0)),
    }


# Warm-up run
warmup = _run_once()
print(f"WARMUP total_ms={warmup['total_ms']:.3f}")

# Measured runs (2 and 3)
run2 = _run_once()
run3 = _run_once()

for idx, run in [(2, run2), (3, run3)]:
    print(
        f"RUN{idx} total_ms={run['total_ms']:.3f} "
        f"forward_ms={run['forward_ms']:.3f} rss_mb={run['rss_mb']:.3f}"
    )

measured = [run2["total_ms"], run3["total_ms"]]
measured_sorted = sorted(measured)

p50 = 0.5 * (measured_sorted[0] + measured_sorted[1])
p95 = measured_sorted[1]
max_rss = max(run2["rss_mb"], run3["rss_mb"])

print(f"P50_total_ms={p50:.3f}")
print(f"P95_total_ms={p95:.3f}")
print(f"MAX_rss_mb={max_rss:.3f}")
PY
