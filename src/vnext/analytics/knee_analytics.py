from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np

try:
    from scipy import signal as _sp_signal

    _HAVE_SCIPY = True
except Exception:
    _sp_signal = None
    _HAVE_SCIPY = False

SCHEMA_VERSION = "knee_analytics_v1"

DEFAULT_MASS_KG = 75.0
DEFAULT_SAMPLE_HZ = 200.0
DEFAULT_LEVER_ARM_M = 0.04
DEFAULT_SMOOTH_WINDOW_S = 0.05


def _use_scipy_lowpass() -> bool:
    return os.environ.get("KNEE_ANALYTICS_USE_SCIPY", "0") == "1"


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if window <= 1:
        return x.astype(np.float32, copy=False)
    pad_left = int(window // 2)
    pad_right = int(window - 1 - pad_left)
    x_pad = np.pad(x, (pad_left, pad_right), mode="edge")
    kernel = (np.ones(int(window), dtype=np.float32) / float(window)).astype(
        np.float32, copy=False
    )
    y = np.convolve(x_pad, kernel, mode="valid")
    return np.asarray(y, dtype=np.float32)


def _lowpass_scipy(x: np.ndarray, sample_hz: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if x.size < 8:
        return x.astype(np.float32, copy=False)
    nyq = float(sample_hz) / 2.0
    cutoff_hz = min(6.0, 0.45 * nyq)
    if cutoff_hz <= 0.0 or not np.isfinite(cutoff_hz):
        return x.astype(np.float32, copy=False)
    b, a = _sp_signal.butter(4, cutoff_hz / nyq, btype="low", analog=False)
    y = _sp_signal.filtfilt(b, a, x.astype(np.float64), method="pad")
    return np.asarray(y, dtype=np.float32)


def compute_knee_moment_from_fz(
    Fz: np.ndarray,
    mass_kg: float,
    sample_hz: float,
    stride_len_m: float | None = None,
) -> Dict[str, Any]:
    fz_n = np.asarray(Fz, dtype=np.float32).reshape(-1)
    if not np.isfinite(fz_n).all():
        raise ValueError("non-finite Fz")

    lever_arm_m = float(DEFAULT_LEVER_ARM_M if stride_len_m is None else stride_len_m)
    if lever_arm_m <= 0.0 or not np.isfinite(lever_arm_m):
        raise ValueError("invalid lever_arm_m")

    mass_kg_f = float(mass_kg)
    if mass_kg_f <= 0.0 or not np.isfinite(mass_kg_f):
        raise ValueError("invalid mass_kg")

    sample_hz_f = float(sample_hz)
    if sample_hz_f <= 0.0 or not np.isfinite(sample_hz_f):
        raise ValueError("invalid sample_hz")

    moment_nm = fz_n * lever_arm_m
    moment_nm_per_kg = moment_nm / mass_kg_f

    window = int(round(sample_hz_f * float(DEFAULT_SMOOTH_WINDOW_S)))
    window = max(1, window)
    use_scipy = bool(_HAVE_SCIPY and _use_scipy_lowpass())
    if use_scipy:
        moment_smoothed = _lowpass_scipy(moment_nm_per_kg, sample_hz_f)
    else:
        moment_smoothed = _moving_average(moment_nm_per_kg, window)

    return {
        "schema_version": SCHEMA_VERSION,
        "mass_kg": float(mass_kg_f),
        "sample_hz": float(sample_hz_f),
        "lever_arm_m": float(lever_arm_m),
        "smooth_window_samples": int(window),
        "filter_kind": "scipy_lowpass" if use_scipy else "moving_average",
        "moment_nm_per_kg": np.asarray(moment_smoothed, dtype=np.float32),
        "moment_nm_per_kg_raw": np.asarray(moment_nm_per_kg, dtype=np.float32),
    }


def summarize_curve(curve: np.ndarray) -> Dict[str, float]:
    x = np.asarray(curve, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "p95": float("nan"),
            "finite_fraction": 0.0,
        }

    finite = np.isfinite(x)
    finite_fraction = float(np.sum(finite)) / float(x.size)
    if not finite.any():
        return {
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "p95": float("nan"),
            "finite_fraction": float(finite_fraction),
        }

    vals = x[finite].astype(np.float64)
    return {
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "p95": float(np.percentile(vals, 95.0)),
        "finite_fraction": float(finite_fraction),
    }
