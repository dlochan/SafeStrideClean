#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

BASELINE_JSON="${IMU_CONTRACT_BASELINE_JSON:-tests/baselines/imu_contract_baseline.json}"
FIXTURE_CSV="${IMU_CONTRACT_FIXTURE_CSV:-tests/fixtures/imu_sample.csv}"
WINDOW_LEN="${IMU_CONTRACT_WINDOW_LEN:-3}"
STRIDE="${IMU_CONTRACT_STRIDE:-1}"

python3 scripts/check_imu_nonregression.py \
  --baseline "$BASELINE_JSON" \
  --fixture "$FIXTURE_CSV" \
  --window-len "$WINDOW_LEN" \
  --stride "$STRIDE"

echo "CI PASS imu_contract"
