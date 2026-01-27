#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
python3 -m unittest -q tests.test_imu_schema_unittest
echo "PASS: imu_schema_smoke"
