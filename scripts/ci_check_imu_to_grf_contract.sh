#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 scripts/check_imu_to_grf_nonregression.py \
  --mode check \
  --baseline tests/baselines/imu_to_grf_contract_baseline.json \
  --print_regen_cmd || exit $?

echo "CI PASS imu_to_grf_contract"
