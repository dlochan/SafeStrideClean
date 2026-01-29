#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m unittest -q tests.test_imu_to_grf_input_adapter

python3 - << 'PY'
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[0]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.adapters.imu_to_grf_input import build_grf_input_from_imu_csv

X_single = build_grf_input_from_imu_csv(Path("tests/fixtures/imu_sample.csv"), window_len=256, stride=256, num_windows=1)
X_batch = build_grf_input_from_imu_csv(Path("tests/fixtures/imu_sample.csv"), window_len=256, stride=4, num_windows=64)

print(f"SHAPE_SINGLE={tuple(X_single.shape)}")
print(f"SHAPE_BATCH={tuple(X_batch.shape)}")
PY
