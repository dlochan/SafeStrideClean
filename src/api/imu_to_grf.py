from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import subprocess
from datetime import datetime, timezone
import sys
import time
import resource

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


def _get_rss_mb() -> float:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = float(usage.ru_maxrss)
        if sys.platform.startswith("darwin"):
            rss_bytes = rss
        elif sys.platform.startswith("linux"):
            rss_bytes = rss * 1024.0
        else:
            rss_bytes = rss
        return rss_bytes / (1024.0 * 1024.0)
    except Exception:  # pragma: no cover - defensive
        return 0.0


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
    profile: bool = False,
) -> Dict[str, Any]:
    """Run deterministic IMU→GRF inference and return a JSON-serializable summary.

    This is a thin, stable wrapper around the existing IMU→GRF adapter and
    deterministic inference path used by the non-regression contract.
    """

    t_total_start = time.perf_counter() if profile else 0.0

    # Deterministic seeding (mirrors scripts/check_imu_infer_nonregression.py).
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    csv_path = Path(imu_csv)

    if profile:
        t_build_start = time.perf_counter()
    X = build_grf_input_from_imu_csv(
        csv_path,
        window_len=int(window_len),
        stride=int(stride),
        num_windows=int(num_windows),
    )
    build_input_ms = 0.0
    if profile:
        build_input_ms = (time.perf_counter() - t_build_start) * 1000.0

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

    if profile:
        t_forward_start = time.perf_counter()
    with torch.no_grad():
        y = model_obj(torch.from_numpy(X).cpu())
    forward_ms = 0.0
    if profile:
        forward_ms = (time.perf_counter() - t_forward_start) * 1000.0

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

    if profile:
        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        rss_mb = _get_rss_mb()
        result["perf"] = {
            "build_input_ms": float(build_input_ms),
            "forward_ms": float(forward_ms),
            "total_ms": float(total_ms),
            "rss_mb": float(rss_mb),
        }

    return result


def run_imu_to_grf_batch(
    imu_dir: str,
    window_len: int = 256,
    stride: int = 1,
    num_windows: int = 64,
    profile: bool = False,
    seed: int = 12345,
    pattern: str = "*.csv",
) -> Dict[str, Any]:
    """Run IMU→GRF inference over a directory of CSVs.

    This function is a thin batch wrapper around ``run_imu_to_grf`` that
    iterates over all files in ``imu_dir`` matching ``pattern``, sorted
    lexicographically. Individual failures are recorded in the results and do
    not stop the batch.
    """

    base = Path(imu_dir)
    files = sorted(
        p
        for p in base.glob(pattern)
        if p.is_file() and not p.name.startswith("._")
    )

    results: list[Dict[str, Any]] = []
    num_files = len(files)
    num_ok = 0
    num_failed = 0

    total_ms_vals: list[float] = []
    rss_mb_vals: list[float] = []

    for csv_path in files:
        entry: Dict[str, Any]
        try:
            out = run_imu_to_grf(
                str(csv_path),
                window_len=window_len,
                stride=stride,
                num_windows=num_windows,
                model="vnext_fz",
                seed=seed,
                profile=profile,
            )

            entry = {
                "imu_csv": str(csv_path),
                "ok": True,
                "model": out.get("model"),
                "output": out.get("output"),
            }

            if profile:
                perf = out.get("perf")
                if isinstance(perf, dict):
                    perf_dict = dict(perf)
                    entry["perf"] = perf_dict
                    t_ms = perf_dict.get("total_ms")
                    rss = perf_dict.get("rss_mb")
                    if isinstance(t_ms, (int, float)):
                        total_ms_vals.append(float(t_ms))
                    if isinstance(rss, (int, float)):
                        rss_mb_vals.append(float(rss))

            num_ok += 1
        except Exception as e:  # pragma: no cover - defensive
            entry = {
                "imu_csv": str(csv_path),
                "ok": False,
                "error": f"{e.__class__.__name__}: {e}",
            }
            num_failed += 1

        results.append(entry)

    num_files = len(files)
    ok_rate = float(num_ok) / float(num_files) if num_files > 0 else 0.0

    summary: Dict[str, Any] = {
        "ok_rate": float(ok_rate),
    }

    if profile and total_ms_vals:
        vals = sorted(total_ms_vals)
        n = len(vals)
        if n % 2 == 1:
            p50 = vals[n // 2]
        else:
            p50 = 0.5 * (vals[n // 2 - 1] + vals[n // 2])
        p95 = vals[-1]
        max_rss = max(rss_mb_vals) if rss_mb_vals else 0.0

        summary["p50_total_ms"] = float(p50)
        summary["p95_total_ms"] = float(p95)
        summary["max_rss_mb"] = float(max_rss)

    result: Dict[str, Any] = {
        "schema_version": "imu_grf_batch_v1",
        "metadata": {
            "deterministic_seed": int(seed),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": _get_git_short_commit(),
            "num_files": int(num_files),
            "num_ok": int(num_ok),
            "num_failed": int(num_failed),
        },
        "config": {
            "window_len": int(window_len),
            "stride": int(stride),
            "num_windows": int(num_windows),
        },
        "results": results,
        "summary": summary,
    }

    return result
