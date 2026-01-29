#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 - << 'PY'
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[0]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.adapters.imu_to_grf_input import build_grf_input_from_imu_csv
from src.vnext.data.imu_schema import get_feature_columns, get_sensor_slices
from src.vnext.models.vnext_fz import VNextFzModel

X = build_grf_input_from_imu_csv(
    Path("tests/fixtures/imu_sample.csv"),
    window_len=256,
    num_windows=64,
    stride=1,
)

if X.dtype != np.float32:
    raise SystemExit(f"Expected float32 input, got {X.dtype}")
if not np.isfinite(X).all():
    raise SystemExit("Non-finite values in input")

feature_cols = get_feature_columns()
sensor_slices = get_sensor_slices(feature_cols)

model = VNextFzModel(
    in_channels=int(X.shape[2]),
    sensor_slices=sensor_slices,
).cpu().eval()

with torch.no_grad():
    y = model(torch.from_numpy(X).cpu())

y_np = y.detach().cpu().numpy()

print(f"INPUT_SHAPE={tuple(X.shape)}")
print(f"OUTPUT_SHAPE={tuple(y_np.shape)}")
print(f"OUTPUT_DTYPE={y_np.dtype}")
print(f"OUTPUT_FINITE={bool(np.isfinite(y_np).all())}")

if not np.isfinite(y_np).all():
    raise SystemExit("Non-finite values in output")
PY
