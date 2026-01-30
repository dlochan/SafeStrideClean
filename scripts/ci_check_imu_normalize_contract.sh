#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "${IMU_NORMALIZE_CONTRACT_SELFTEST:-0}" == "1" ]]; then
  python3 scripts/check_imu_normalize_nonregression.py \
    --mode check \
    --baseline tests/baselines/imu_normalize_contract_baseline.json \
    --self_test \
    --self_test_case all \
    --print_regen_cmd
  exit $?
fi

python3 scripts/check_imu_normalize_nonregression.py \
  --mode check \
  --baseline tests/baselines/imu_normalize_contract_baseline.json \
  --print_regen_cmd

echo "CI PASS imu_normalize_contract"
