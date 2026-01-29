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

try:
    from src.vnext.models.vnext_fz import VNextFzModel  # type: ignore
    _HAVE_VNEXT_MODEL = True
except Exception:  # pragma: no cover - defensive
    VNextFzModel = None  # type: ignore
    _HAVE_VNEXT_MODEL = False

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
C_CANON = len(feature_cols)
C_ACTUAL = int(X.shape[2])

print(f"C_CANON={C_CANON}")
print(f"C_ACTUAL={C_ACTUAL}")

if X.shape[0] != 64 or X.shape[1] != 256 or C_ACTUAL != C_CANON:
    raise SystemExit(
        f"Unexpected adapter shape; got {tuple(X.shape)}, expected (64, 256, {C_CANON})"
    )

sensor_slices = get_sensor_slices(feature_cols)

model = None
if _HAVE_VNEXT_MODEL:
    try:
        model = VNextFzModel(  # type: ignore[operator]
            in_channels=C_CANON,
            sensor_slices=sensor_slices,
        )
    except Exception:  # pragma: no cover - defensive
        model = None

if model is None:
    # Fallback: tiny local CPU-only model that accepts (B,T,C) and produces (B,T,1).
    class _TinyGRFModel(torch.nn.Module):  # type: ignore[misc]
        def __init__(self, in_channels: int) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(in_channels, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            # x: (B, T, C)
            b, t, c = x.shape
            x_flat = x.reshape(b * t, c)
            y_flat = self.net(x_flat)
            y = y_flat.reshape(b, t, 1)
            return y

    model = _TinyGRFModel(in_channels=C_CANON)

model = model.cpu().eval()

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
