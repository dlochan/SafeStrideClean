from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

EPS = 1e-8


def _safe_corr_1d(a: np.ndarray, b: np.ndarray) -> float:
    """Safe Pearson correlation for 1D arrays.

    Returns NaN if arrays are empty, mismatched, or nearly constant.
    """

    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    if x.size == 0 or y.size != x.size:
        return float("nan")
    sx = float(np.std(x))
    sy = float(np.std(y))
    if sx < EPS or sy < EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> Tuple[int, float]:
    """Return (lag, corr) where positive lag means b is delayed relative to a.

    This mirrors the Fz analyzer's behavior but is kept self-contained here so
    the 3D analyzer does not depend on the Fz-only tooling.
    """

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size or a.size == 0:
        return 0, float("nan")

    def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.size != y.size or x.size == 0:
            return float("nan")
        sx = float(np.std(x))
        sy = float(np.std(y))
        if sx < EPS or sy < EPS:
            return float("nan")
        return float(np.corrcoef(x, y)[0, 1])

    best_lag = 0
    best_corr = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            aa = a[-lag:]
            bb = b[: a.size + lag]
        elif lag > 0:
            aa = a[: a.size - lag]
            bb = b[lag:]
        else:
            aa = a
            bb = b
        r = _safe_corr(aa, bb)
        if not np.isfinite(r):
            continue
        if r > best_corr:
            best_corr = r
            best_lag = lag
    if best_corr == -np.inf:
        return 0, float("nan")
    return best_lag, float(best_corr)


def analyze_3d_run(run_dir: Path, out_dir: Path, preds_suffix: str | None) -> Dict[str, Any]:
    """Minimal 3D GRF analyzer.

    - Loads 3D GRF prediction NPZ from run_dir/eval/preds.
    - Computes per-axis RMSE + basic magnitude/constant checks.
    - Computes an Fz-only temporal lag statistic using cross-correlation.
    - Writes analysis/3d_metrics_summary.json into out_dir.
    """

    if not run_dir.exists() or not run_dir.is_dir():
        raise SystemExit(f"run_dir does not exist or is not a directory: {run_dir}")

    preds_dir = run_dir / "eval" / "preds"
    if not preds_dir.exists() or not preds_dir.is_dir():
        raise SystemExit(f"preds directory not found: {preds_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the canonical grf3d_windows_pred_truth*.npz naming, but fall back
    # to the first NPZ in the preds directory for robustness.
    npz_path: Path | None = None
    if preds_suffix:
        candidate = preds_dir / f"grf3d_windows_pred_truth_{preds_suffix}.npz"
        if candidate.exists():
            npz_path = candidate
    if npz_path is None:
        candidate = preds_dir / "grf3d_windows_pred_truth.npz"
        if candidate.exists():
            npz_path = candidate
    if npz_path is None:
        # Fallback: any NPZ in preds dir.
        for p in sorted(preds_dir.glob("*.npz")):
            npz_path = p
            break

    if npz_path is None:
        raise SystemExit(f"No NPZ prediction export found under {preds_dir}")

    data = np.load(npz_path, allow_pickle=True)

    if "y_true" not in data or "y_pred" not in data:
        raise SystemExit(f"NPZ missing required arrays 'y_true'/'y_pred': {npz_path}")

    y_true = np.asarray(data["y_true"], dtype=np.float64)
    y_pred = np.asarray(data["y_pred"], dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise SystemExit(f"Shape mismatch in NPZ: y_true{y_true.shape} vs y_pred{y_pred.shape}")

    if y_true.ndim == 2:
        # (N, T) → treat as single-axis signal (Fz-only) for completeness.
        y_true = y_true[..., None]
        y_pred = y_pred[..., None]

    if y_true.ndim != 3:
        raise SystemExit(f"Expected y_true/y_pred with shape (N, T, D); got {y_true.shape}")

    n_windows, window_len, n_axes = y_true.shape
    if n_windows == 0 or window_len == 0 or n_axes == 0:
        raise SystemExit("NPZ contains empty arrays; cannot analyze")

    axis_names: List[str]
    if n_axes == 1:
        axis_names = ["Fz"]
    elif n_axes == 2:
        axis_names = ["Fx", "Fz"]
    else:
        axis_names = ["Fx", "Fy", "Fz"][:n_axes]

    axis_summaries: Dict[str, Any] = {}
    gate_reasons: List[str] = []
    corr_fail_axes: List[str] = []

    for axis_idx, axis_name in enumerate(axis_names):
        # Windowed views for this axis
        yt_win = y_true[:, :, axis_idx]
        yp_win = y_pred[:, :, axis_idx]

        # Core error statistics in the space provided by eval_vnext.py
        err = yp_win - yt_win
        rmse = float(np.sqrt(np.mean(err**2)))
        std_pred = float(np.std(yp_win))
        std_true = float(np.std(yt_win))
        max_abs_pred = float(np.max(np.abs(yp_win)))
        max_abs_true = float(np.max(np.abs(yt_win)))
        ratio = max_abs_pred / (max_abs_true + EPS) if max_abs_true > 0 else float("inf")

        # Flatten across windows/time for scalar stats and correlation.
        yt = yt_win.reshape(-1)
        yp = yp_win.reshape(-1)
        mean_true = float(np.mean(yt))
        mean_pred = float(np.mean(yp))
        median_abs_true = float(np.median(np.abs(yt)))
        median_abs_pred = float(np.median(np.abs(yp)))
        min_true = float(np.min(yt))
        max_true = float(np.max(yt))
        min_pred = float(np.min(yp))
        max_pred = float(np.max(yp))
        corr = _safe_corr_1d(yt, yp)

        # Detect space: normalized (~O(1)) vs Newton (~O(10^2-10^3)).
        if std_true < 10.0 and median_abs_true < 10.0:
            scale = "normalized"
            rmse_threshold = 5.0
        else:
            scale = "newton"
            rmse_threshold = 500.0

        axis_info: Dict[str, Any] = {
            "rmse": rmse,
            "std_pred": std_pred,
            "std_true": std_true,
            "max_abs_pred": max_abs_pred,
            "max_abs_true": max_abs_true,
            "max_abs_ratio": ratio,
            "constant_pred": bool(std_pred < 1e-6),
            "mean_true": mean_true,
            "mean_pred": mean_pred,
            "median_abs_true": median_abs_true,
            "median_abs_pred": median_abs_pred,
            "min_true": min_true,
            "max_true": max_true,
            "min_pred": min_pred,
            "max_pred": max_pred,
            "corr": corr,
            "scale": scale,
            "rmse_threshold": rmse_threshold,
        }

        # RMSE-based gate: fail if above the scale-appropriate threshold.
        if rmse > rmse_threshold:
            gate_reasons.append(
                f"{axis_name}: RMSE {rmse:.2f} exceeds threshold {rmse_threshold:.2f} in {scale} space"
            )

        # Correlation-based gate: track axes with low/undefined correlation.
        corr_min_abs = 0.3
        axis_info["corr_min_abs"] = corr_min_abs
        corr_ok = bool(np.isfinite(corr) and abs(corr) >= corr_min_abs)
        axis_info["corr_ok"] = corr_ok
        if not corr_ok:
            corr_fail_axes.append(axis_name)

        # Fz temporal lag: only for the vertical component.
        if axis_name.lower().endswith("z"):
            lags: List[int] = []
            max_lag = min(20, window_len // 4) if window_len > 0 else 0
            for i in range(n_windows):
                lag, _ = _best_lag(yt_win[i], yp_win[i], max_lag=max_lag)
                lags.append(int(lag))

            if lags:
                lag_arr = np.asarray(lags, dtype=np.float64)
                med_signed_raw = float(np.median(lag_arr))
                med_abs_raw = float(np.median(np.abs(lag_arr)))
                # Residual lags after correcting for global median signed lag
                residual = lag_arr - med_signed_raw
                med_abs = float(np.median(np.abs(residual)))
            else:
                med_signed_raw = 0.0
                med_abs_raw = 0.0
                med_abs = 0.0

            lag_ok = med_abs <= 2.0
            axis_info["temporal_lag"] = {
                "status": "PASS" if lag_ok else "FAIL",
                "median_abs_lag_samples_residual": med_abs,
                "median_abs_lag_samples_raw": med_abs_raw,
                "median_signed_lag_samples_raw": med_signed_raw,
                "max_lag_checked": int(max_lag),
            }
            if not lag_ok:
                gate_reasons.append(
                    "Fz temporal lag FAILED: "
                    f"median_abs_residual={med_abs:.2f} samples (max_lag_checked={max_lag})"
                )

        axis_summaries[axis_name] = axis_info

    # Global correlation gate: fail if *all* axes are low-correlation.
    if corr_fail_axes and len(corr_fail_axes) == len(axis_names):
        gate_reasons.append(
            "Correlation near zero on all axes; low-correlation axes="
            + ",".join(sorted(corr_fail_axes))
        )

    gate_ok = not gate_reasons

    summary: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "npz_path": str(npz_path),
        "num_windows": int(n_windows),
        "window_len": int(window_len),
        "axis_summaries": axis_summaries,
    }

    summary["gate"] = {
        "status": "PASS" if gate_ok else "FAIL",
        "reasons": gate_reasons,
        "rmse_thresholds": {"normalized": 5.0, "newton": 500.0},
        "corr_min_abs": 0.3,
    }

    out_path = out_dir / "3d_metrics_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Minimal 3D GRF analysis for vNext")
    ap.add_argument("--run-dir", required=True, help="Run directory containing eval/preds NPZ exports")
    ap.add_argument(
        "--preds-suffix",
        default=None,
        help=(
            "Optional suffix matching eval_vnext.py --preds-suffix; "
            "expects files named grf3d_windows_pred_truth_<suffix>.npz."
        ),
    )
    ap.add_argument(
        "--out-dir",
        required=True,
        help="Output directory for 3D metrics summary JSON",
    )
    args = ap.parse_args()

    summary = analyze_3d_run(Path(args.run_dir), Path(args.out_dir), args.preds_suffix)
    # Print a compact JSON summary to stdout for logs.
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
