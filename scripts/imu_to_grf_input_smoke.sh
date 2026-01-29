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

X = build_grf_input_from_imu_csv(Path("tests/fixtures/imu_sample.csv"), window_len=3, stride=3)
print(f"SHAPE={tuple(X.shape)} dtype={X.dtype} finite={bool(np.isfinite(X).all())}")
PY
