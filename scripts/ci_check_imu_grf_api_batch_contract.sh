#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

python3 -m unittest -q tests.contracts.test_imu_grf_api_batch_contract

echo "CI PASS imu_grf_api_batch_contract"
