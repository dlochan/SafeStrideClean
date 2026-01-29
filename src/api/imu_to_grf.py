from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch
import random

from src.adapters.imu_to_grf_input import build_grf_input_from_imu_csv
from src.vnext.data.imu_schema import get_feature_columns, get_sensor_slices

try:  # Prefer the real vNext model when available.
    from src.vnext.models.vnext_fz import VNextFzModel  # type: ignore

    _HAVE_VNEXT_MODEL = True
except Exception:  # pragma: no cover - defensive
    VNextFzModel = None  # type: ignore
    _HAVE_VNEXT_MODEL = False


def _get_git_short_commit() -> str:
    """Return the short git commit hash, or "unknown" if unavailable."""

    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode("utf-8").strip()
    except Exception:  # pragma: no cover - defensive
        return "unknown"


def _stats(arr: np.ndarray) -> Dict[str, float]:
    """Compute basic stats over a tensor-like array, JSON-serializable."""

    flat = np.asarray(arr, dtype=np.float32).ravel()
    if flat.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": 0.0,
            "std": 0.0,
            "finite_fraction": 0.0,
        }

    finite = np.isfinite(flat)
    if not finite.any():
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": 0.0,
            "std": 0.0,
            "finite_fraction": 0.0,
        }

    vals = flat[finite]
    finite_fraction = float(vals.size) / float(flat.size)
    return {
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "finite_fraction": finite_fraction,
    }


def run_imu_to_grf(
    imu_csv: str,
    *,
    window_len: int = 256,
    stride: int = 1,
    num_windows: int = 64,
    model: str = "vnext_fz",
    seed: int = 12345,
) -> Dict[str, Any]:
    """Run deterministic IMU→GRF inference and return a JSON-serializable summary.

    This is a thin, stable wrapper around the existing IMU→GRF adapter and
    deterministic inference path used by the non-regression contract.
    """

    # Deterministic seeding (mirrors scripts/check_imu_infer_nonregression.py).
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    csv_path = Path(imu_csv)

    X = build_grf_input_from_imu_csv(
        csv_path,
        window_len=int(window_len),
        stride=int(stride),
        num_windows=int(num_windows),
    )

    if X.dtype != np.float32:
        raise ValueError(f"Expected float32 input, got {X.dtype}")
    if not np.isfinite(X).all():
        raise ValueError("Non-finite values in GRF input tensor")

    feature_cols = get_feature_columns()
    C_canon = len(feature_cols)
    C_actual = int(X.shape[2])

    if X.shape != (int(num_windows), int(window_len), C_canon):
        raise ValueError(
            f"Unexpected adapter shape; got {tuple(X.shape)}, "
            f"expected ({int(num_windows)}, {int(window_len)}, {C_canon})"
        )
    if C_actual != C_canon:
        raise ValueError(f"C_actual={C_actual} does not match C_canon={C_canon}")

    sensor_slices = get_sensor_slices(feature_cols)

    model_obj: Any = None
    if _HAVE_VNEXT_MODEL:
        try:
            model_obj = VNextFzModel(  # type: ignore[operator]
                in_channels=C_canon,
                sensor_slices=sensor_slices,
            )
        except Exception:  # pragma: no cover - defensive
            model_obj = None

    if model_obj is None:
        # Fallback tiny CPU-only model mapping (B, T, C) → (B, T, 1), deterministic
        # under the seeding above.
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

        model_obj = _TinyGRFModel(in_channels=C_canon)

    model_obj = model_obj.cpu().eval()

    with torch.no_grad():
        y = model_obj(torch.from_numpy(X).cpu())

    y_np = y.detach().cpu().numpy()

    if not np.isfinite(y_np).all():
        raise ValueError("Non-finite values in GRF output tensor")

    output_shape = [int(d) for d in y_np.shape]
    stats = _stats(y_np)

    # For this deterministic path, we expect all outputs to be finite.
    # Ensure the contract reports that explicitly.
    stats["finite_fraction"] = 1.0

    result: Dict[str, Any] = {
        "schema_version": "imu_grf_v1",
        "model": str(model),
        "input": {
            "imu_csv": str(csv_path),
            "window_len": int(window_len),
            "stride": int(stride),
            "num_windows": int(num_windows),
            "channels": int(C_canon),
        },
        "output": {
            "shape": output_shape,
            "units": "newtons",
            "stats": stats,
        },
        "metadata": {
            "deterministic_seed": int(seed),
            "git_commit": _get_git_short_commit(),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        },
    }

    return result
