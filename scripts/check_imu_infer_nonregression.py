from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import sys

import numpy as np
import torch
import random

_REPO_ROOT = Path(__file__).resolve().parents[1]
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


def _run_inference(
    *,
    fixture: str,
    window_len: int,
    num_windows: int,
    stride: int,
) -> Dict[str, Any]:
    # Ensure deterministic weights/outputs across runs for contract checking.
    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    X = build_grf_input_from_imu_csv(
        Path(fixture),
        window_len=int(window_len),
        num_windows=int(num_windows),
        stride=int(stride),
    )

    if X.dtype != np.float32:
        raise SystemExit(f"Expected float32 input, got {X.dtype}")
    if not np.isfinite(X).all():
        raise SystemExit("Non-finite values in input")

    feature_cols = get_feature_columns()
    C_canon = len(feature_cols)
    C_actual = int(X.shape[2])

    if X.shape != (num_windows, window_len, C_canon):
        raise SystemExit(
            f"30: Unexpected adapter shape; got {tuple(X.shape)}, expected ({num_windows}, {window_len}, {C_canon})"
        )

    if C_actual != C_canon:
        raise SystemExit(f"31: C_actual={C_actual} does not match C_canon={C_canon}")

    sensor_slices = get_sensor_slices(feature_cols)

    model = None
    if _HAVE_VNEXT_MODEL:
        try:
            model = VNextFzModel(  # type: ignore[operator]
                in_channels=C_canon,
                sensor_slices=sensor_slices,
            )
        except Exception:  # pragma: no cover - defensive
            model = None

    if model is None:
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

        model = _TinyGRFModel(in_channels=C_canon)

    model = model.cpu().eval()

    with torch.no_grad():
        y = model(torch.from_numpy(X).cpu())

    y_np = y.detach().cpu().numpy()

    if not np.isfinite(y_np).all():
        raise SystemExit("32: Non-finite values in output")

    return {"X": X, "y": y_np, "C_canon": C_canon}


def _stats(arr: np.ndarray) -> Dict[str, float]:
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
    return {
        "min": float(vals.min()),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "finite_fraction": float(vals.size) / float(flat.size),
    }


def compute_infer_contract(
    *,
    fixture: str,
    window_len: int,
    num_windows: int,
    stride: int,
) -> Dict[str, Any]:
    out = _run_inference(
        fixture=fixture,
        window_len=window_len,
        num_windows=num_windows,
        stride=stride,
    )
    X = out["X"]
    y = out["y"]
    C_canon = int(out["C_canon"])

    return {
        "input_shape": list(map(int, X.shape)),
        "output_shape": list(map(int, y.shape)),
        "C_canon": C_canon,
        "input_stats": _stats(X),
        "output_stats": _stats(y),
    }


def _check_stats(
    *,
    tag: str,
    actual: Dict[str, float],
    baseline: Dict[str, float],
    mean_tol: float,
    std_tol: float,
    base_exit: int,
) -> None:
    # Finite fraction must be exactly 1.0 and match baseline.
    af = float(actual.get("finite_fraction", 0.0))
    bf = float(baseline.get("finite_fraction", 0.0))
    if af != 1.0 or bf != 1.0:
        raise SystemExit(f"{base_exit}: {tag}.finite_fraction != 1.0 actual={af} baseline={bf}")

    a_mean = float(actual.get("mean", 0.0))
    b_mean = float(baseline.get("mean", 0.0))
    d_mean = abs(a_mean - b_mean)
    if d_mean > mean_tol:
        raise SystemExit(
            f"{base_exit+1}: {tag}.mean drift too large actual={a_mean} baseline={b_mean} diff={d_mean} tol={mean_tol}"
        )

    a_std = float(actual.get("std", 0.0))
    b_std = float(baseline.get("std", 0.0))
    d_std = abs(a_std - b_std)
    if d_std > std_tol:
        raise SystemExit(
            f"{base_exit+2}: {tag}.std drift too large actual={a_std} baseline={b_std} diff={d_std} tol={std_tol}"
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--fixture", default="tests/fixtures/imu_sample.csv")
    ap.add_argument("--window-len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--num-windows", type=int, default=64)
    ap.add_argument("--mode", choices=["compute", "check"], default="check")
    args = ap.parse_args()

    baseline_path = Path(args.baseline)

    if args.mode == "compute":
        summary = compute_infer_contract(
            fixture=args.fixture,
            window_len=int(args.window_len),
            num_windows=int(args.num_windows),
            stride=int(args.stride),
        )
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Wrote IMU infer contract baseline to {baseline_path}")
        return 0

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    actual = compute_infer_contract(
        fixture=args.fixture,
        window_len=int(args.window_len),
        num_windows=int(args.num_windows),
        stride=int(args.stride),
    )

    if actual.get("input_shape") != baseline.get("input_shape"):
        raise SystemExit(
            f"33: input_shape mismatch actual={actual.get('input_shape')} baseline={baseline.get('input_shape')}"
        )
    if actual.get("output_shape") != baseline.get("output_shape"):
        raise SystemExit(
            f"34: output_shape mismatch actual={actual.get('output_shape')} baseline={baseline.get('output_shape')}"
        )
    if int(actual.get("C_canon", 0)) != int(baseline.get("C_canon", 0)):
        raise SystemExit(
            f"35: C_canon mismatch actual={actual.get('C_canon')} baseline={baseline.get('C_canon')}"
        )

    mean_tol = 1e-4
    std_tol = 1e-4

    _check_stats(
        tag="input_stats",
        actual=dict(actual.get("input_stats", {})),
        baseline=dict(baseline.get("input_stats", {})),
        mean_tol=mean_tol,
        std_tol=std_tol,
        base_exit=36,
    )
    _check_stats(
        tag="output_stats",
        actual=dict(actual.get("output_stats", {})),
        baseline=dict(baseline.get("output_stats", {})),
        mean_tol=mean_tol,
        std_tol=std_tol,
        base_exit=40,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
