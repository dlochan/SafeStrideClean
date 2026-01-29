#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [ -z "${ROOT}" ]; then
  echo "ERROR: not inside a git repo"
  exit 2
fi

if [ "$(pwd)" != "${ROOT}" ]; then
  echo "NOTE: changing directory to repo root: ${ROOT}"
fi
cd "${ROOT}"

echo "=== IMU DEMO ==="
echo "HEAD=$(git rev-parse --short HEAD)"
echo "BRANCH=$(git branch --show-current)"

echo "--- imu_schema_smoke ---"
bash scripts/imu_schema_smoke.sh
echo "PASS: imu_schema_smoke"

echo "--- imu_features_smoke ---"
bash scripts/imu_features_smoke.sh
echo "PASS: imu_features_smoke"

echo "--- ci_check_imu_contract ---"
bash scripts/ci_check_imu_contract.sh
echo "PASS: ci_check_imu_contract"

echo "--- imu_ingest_smoke ---"
bash scripts/imu_ingest_smoke.sh
echo "PASS: imu_ingest_smoke"

echo "--- imu_to_grf_infer_smoke ---"
bash scripts/imu_to_grf_infer_smoke.sh
echo "PASS: imu_to_grf_infer_smoke"

echo "IMU DEMO: PASS"
