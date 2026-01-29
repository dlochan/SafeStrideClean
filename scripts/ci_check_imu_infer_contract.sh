#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

BASELINE_JSON="${IMU_INFER_BASELINE_JSON:-tests/baselines/imu_infer_contract_baseline.json}"
FIXTURE_CSV="${IMU_INFER_FIXTURE_CSV:-tests/fixtures/imu_sample.csv}"
WINDOW_LEN="${IMU_INFER_WINDOW_LEN:-256}"
STRIDE="${IMU_INFER_STRIDE:-1}"
NUM_WINDOWS="${IMU_INFER_NUM_WINDOWS:-64}"

# Run the inference smoke first (builds input + forward pass).
bash scripts/imu_to_grf_infer_smoke.sh

# Then enforce the non-regression contract.
python3 scripts/check_imu_infer_nonregression.py \
  --baseline "$BASELINE_JSON" \
  --fixture "$FIXTURE_CSV" \
  --window-len "$WINDOW_LEN" \
  --stride "$STRIDE" \
  --num-windows "$NUM_WINDOWS"

echo "CI PASS imu_infer_contract"
