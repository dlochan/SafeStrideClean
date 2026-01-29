#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

P95_TOTAL_MS_BUDGET=250
MAX_RSS_MB_BUDGET=900

BENCH_OUT="$(bash scripts/bench_imu_to_grf.sh)"

echo "${BENCH_OUT}"

P95_TOTAL_MS_BUDGET="${P95_TOTAL_MS_BUDGET}" \
MAX_RSS_MB_BUDGET="${MAX_RSS_MB_BUDGET}" \
BENCH_OUT="${BENCH_OUT}" \
python3 - << 'PY'
from __future__ import annotations

import os
import sys

text = os.environ.get("BENCH_OUT", "")

p95 = None
max_rss = None
for line in text.splitlines():
    if line.startswith("P95_total_ms="):
        try:
            p95 = float(line.split("=", 1)[1])
        except ValueError:
            p95 = None
    elif line.startswith("MAX_rss_mb="):
        try:
            max_rss = float(line.split("=", 1)[1])
        except ValueError:
            max_rss = None

if p95 is None or max_rss is None:
    print("CI FAIL imu_grf_perf: missing metrics from bench_imu_to_grf.sh", file=sys.stderr)
    sys.exit(1)

p95_budget = float(os.environ.get("P95_TOTAL_MS_BUDGET", "250"))
max_rss_budget = float(os.environ.get("MAX_RSS_MB_BUDGET", "900"))

if p95 > p95_budget or max_rss > max_rss_budget:
    print(
        f"CI FAIL imu_grf_perf: P95_total_ms={p95:.3f} (budget={p95_budget:.3f}), "
        f"MAX_rss_mb={max_rss:.3f} (budget={max_rss_budget:.3f})"
    )
    sys.exit(1)

print(
    f"CI PASS imu_grf_perf: P95_total_ms={p95:.3f} (budget={p95_budget:.3f}), "
    f"MAX_rss_mb={max_rss:.3f} (budget={max_rss_budget:.3f})"
)
PY
