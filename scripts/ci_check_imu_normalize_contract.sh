#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 scripts/check_imu_normalize_nonregression.py \
  --mode check \
  --baseline tests/baselines/imu_normalize_contract_baseline.json

echo "CI PASS imu_normalize_contract"
