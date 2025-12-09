# scripts/make_features.py
import argparse
import json
import os
import sys
import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from src.features_dual import build_dual_features


def write_parquet(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Try pyarrow then fastparquet
    last_err = None
    for engine in ("pyarrow", "fastparquet"):
        try:
            df.to_parquet(path, engine=engine, index=False)
            return
        except Exception as e:
            last_err = e
    raise SystemExit(f"Failed to write parquet at {path}: {last_err}\nInstall 'pyarrow' or 'fastparquet'.")


def align_y_to_windows(grf: pd.DataFrame, t: pd.Series, half_window_s: float) -> pd.DataFrame:
    grf_sorted = grf.sort_values("time_s").reset_index(drop=True)
    tg = grf_sorted["time_s"].to_numpy(dtype=float)
    centers = t.to_numpy(dtype=float)
    idxs = np.searchsorted(tg, centers)
    # clamp indices to nearest neighbor
    chosen = []
    for i, c in enumerate(centers):
        candidates = []
        if idxs[i] < len(tg):
            candidates.append(idxs[i])
        if idxs[i] > 0:
            candidates.append(idxs[i]-1)
        if not candidates:
            chosen.append(None)
            continue
        # pick closer time
        best = min(candidates, key=lambda j: abs(tg[j] - c))
        # enforce tolerance: must be within half_window_s
        if abs(tg[best] - c) <= half_window_s + 1e-9:
            chosen.append(best)
        else:
            chosen.append(None)
    # Build output
    out = pd.DataFrame({"time_s": centers})
    out["Fx_N"] = [grf_sorted["Fx_N"].iloc[j] if j is not None else np.nan for j in chosen]
    out["Fy_N"] = [grf_sorted["Fy_N"].iloc[j] if j is not None else np.nan for j in chosen]
    out["Fz_N"] = [grf_sorted["Fz_N"].iloc[j] if j is not None else np.nan for j in chosen]
    return out


def _setup_logging(default_name: str, user_log: str | None) -> None:
    log_path = None
    if user_log:
        log_path = user_log
    else:
        logs_root = os.getenv("SAFESTRIDE_LOGS_ROOT", r"C:\\Users\\locha\\Documents\\safestride\\logs")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(Path(logs_root) / f"{default_name}_{ts}.log")
    try:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        f = open(log_path, "a", encoding="utf-8")
    except Exception:
        return
    class _Tee:
        def __init__(self, fobj):
            self.f = fobj
            self._stdout = sys.stdout
            self._stderr = sys.stderr
        def write(self, s):
            try:
                self.f.write(s)
            except Exception:
                pass
            return self._stdout.write(s)
        def flush(self):
            try:
                self.f.flush()
            except Exception:
                pass
            return self._stdout.flush()
    tee = _Tee(f)
    sys.stdout = tee
    sys.stderr = tee


def main():
    ap = argparse.ArgumentParser(description="Build dual-IMU features and labels parquet files.")
    ap.add_argument("--imu_csv", required=True)
    ap.add_argument("--grf_csv", required=True)
    ap.add_argument("--window_ms", type=int, required=True)
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--log_file", default=None, help="Optional log file to tee output; otherwise logs to LOGS_ROOT")
    args = ap.parse_args()

    # Setup logging
    stem = Path(args.imu_csv).stem if args.imu_csv else "features"
    _setup_logging(f"features_{stem}_w{args.window_ms}", args.log_file)

    imu = pd.read_csv(args.imu_csv)
    grf = pd.read_csv(args.grf_csv)

    # Build features
    X, t = build_dual_features(imu, fs=args.fs, window_ms=args.window_ms)

    # Align labels (Fx_N, Fy_N, Fz_N) to window centers
    half_window_s = 0.5 * args.window_ms / 1000.0
    if not {"Fx_N", "Fy_N", "Fz_N"}.issubset(set(grf.columns)):
        raise SystemExit("GRF CSV must include Fx_N, Fy_N, Fz_N")

    y = align_y_to_windows(grf[["time_s", "Fx_N", "Fy_N", "Fz_N"]].copy(), t, half_window_s)
    # Drop rows with missing labels due to tolerance
    valid = ~y[["Fx_N", "Fy_N", "Fz_N"]].isna().any(axis=1)
    X = X.loc[valid].reset_index(drop=True)
    y = y.loc[valid, ["Fx_N", "Fy_N", "Fz_N"]].reset_index(drop=True)
    t = t.loc[valid].reset_index(drop=True)

    # Sanity
    assert len(X) == len(y) == len(t)
    assert not np.isnan(X.values).any()
    assert not np.isnan(y.values).any()

    out_prefix = Path(args.out_prefix)
    x_path = out_prefix.with_name(out_prefix.name + "_X").with_suffix(".parquet")
    y_path = out_prefix.with_name(out_prefix.name + "_y").with_suffix(".parquet")
    m_path = out_prefix.with_name(out_prefix.name + "_meta").with_suffix(".json")

    write_parquet(X, x_path)
    write_parquet(y, y_path)

    # Meta JSON
    meta = {
        "fs_hz": args.fs,
        "window_ms": args.window_ms,
        "n_windows": int(len(X)),
        "n_features": int(X.shape[1]),
        "y_columns": ["Fx_N", "Fy_N", "Fz_N"],
        "time_center_min": float(t.min()) if len(t) else None,
        "time_center_max": float(t.max()) if len(t) else None,
    }
    # If GRF file had an accompanying lag json, record it
    lag_json_guess = Path(str(Path(args.grf_csv)) + ".lag.json")
    if lag_json_guess.exists():
        meta["lag_json"] = str(lag_json_guess)
    try:
        m_path.parent.mkdir(parents=True, exist_ok=True)
        with open(m_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    except Exception as e:
        raise SystemExit(f"Failed to write meta JSON: {e}")

    print(f"[OK] wrote {x_path} and {y_path}\n[OK] meta {m_path}")


if __name__ == "__main__":
    main()
