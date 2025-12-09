# scripts/auto_align_shift_grf.py
"""
Auto-detect lag between IMU (acc magnitude) and GRF (Fz_N) using cross-correlation,
then shift GRF by samples to minimize lag. If the first attempt makes it worse,
it automatically tries the opposite shift direction and keeps the better one.

Usage:
  python scripts/auto_align_shift_grf.py ^
    --imu data\working\gt_AB01_cutting_leftfast_imu.csv ^
    --grf_in data\working\gt_AB01_cutting_leftfast_grf.csv ^
    --grf_out data\working\gt_AB01_cutting_leftfast_grf_shifted.csv ^
    --fs 200
"""

import argparse
import os
import sys
import datetime
from pathlib import Path
import json
import numpy as np
import pandas as pd

# NEW: pick accel triplet from dual-sensor (thigh/shank) or legacy single-sensor schema
def _pick_imu_triplet(imu_df):
    """
    Returns (ax_col, ay_col, az_col) from whichever schema is available:
      - dual thigh:  ax_thigh, ay_thigh, az_thigh  (preferred)
      - dual shank:  ax_shank, ay_shank, az_shank
      - legacy:      ax, ay, az
    """
    candidates = [
        ("ax_thigh", "ay_thigh", "az_thigh"),
        ("ax_shank", "ay_shank", "az_shank"),
        ("ax", "ay", "az"),
    ]
    for triplet in candidates:
        if all(c in imu_df.columns for c in triplet):
            return triplet
    raise KeyError(
        f"No IMU accel triplet found. Needed one of {candidates}. "
        f"Got: {list(imu_df.columns)}"
    )

def compute_lag_samples(imu_csv, grf_csv, fs):
    imu = pd.read_csv(imu_csv)
    grf = pd.read_csv(grf_csv)

    # CHANGED: only require time_s in IMU; accel triplet is resolved dynamically
    assert "time_s" in imu.columns, f"time_s missing in {imu_csv}"
    assert "time_s" in grf.columns and "Fz_N" in grf.columns, f"time_s/Fz_N missing in {grf_csv}"

    # NEW: choose available accel triplet (thigh -> shank -> legacy)
    ax_c, ay_c, az_c = _pick_imu_triplet(imu)

    acc_mag = np.sqrt(imu[ax_c]**2 + imu[ay_c]**2 + imu[az_c]**2).to_numpy()
    fz = grf["Fz_N"].fillna(0.0).to_numpy()

    n = min(len(acc_mag), len(fz))
    acc = acc_mag[:n] - np.mean(acc_mag[:n])
    f   = fz[:n]      - np.mean(fz[:n])

    xc = np.correlate(acc, f, mode="full")
    lags = np.arange(-n+1, n)
    best = int(lags[np.argmax(xc)])
    ms = 1000.0 * best / fs
    return best, ms, n

def shift_series_by_samples(y: np.ndarray, k: int) -> np.ndarray:
    """Positive k delays y (pads zeros at start). Negative k advances y (pads zeros at end)."""
    if k == 0:
        return y.copy()
    if k > 0:
        return np.concatenate([np.zeros(k), y[:-k]])
    k = -k
    return np.concatenate([y[k:], np.zeros(k)])

def write_shifted(grf_in, grf_out, shift_samples, force_cols=("Fx_N","Fy_N","Fz_N")):
    df = pd.read_csv(grf_in)
    if "time_s" not in df.columns:
        raise SystemExit(f"time_s missing in {grf_in}")
    cols = [c for c in force_cols if c in df.columns]
    if not cols:
        raise SystemExit(f"No force columns found in {grf_in}")

    out = df.copy()
    for c in cols:
        y = out[c].fillna(0.0).to_numpy()
        out[c] = shift_series_by_samples(y, shift_samples)
    Path(grf_out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(grf_out, index=False)

def choose_shift_and_apply(imu_csv: str, grf_in: str, fs: float, out_path: Path, max_ms: float = 10.0):
    """Compute lag, attempt both directions, cap applied shift to <= max_ms, write shifted CSV and JSON summary.
    Returns: dict summary, final_out Path
    """
    lag0, ms0, n = compute_lag_samples(imu_csv, grf_in, fs)

    # Try A and B (opposite directions)
    shift_a = -lag0
    tmp_a = out_path.with_suffix(".tmpA.csv")
    write_shifted(grf_in, str(tmp_a), shift_a)
    lag_a, ms_a, _ = compute_lag_samples(imu_csv, str(tmp_a), fs)

    shift_b = lag0
    tmp_b = out_path.with_suffix(".tmpB.csv")
    write_shifted(grf_in, str(tmp_b), shift_b)
    lag_b, ms_b, _ = compute_lag_samples(imu_csv, str(tmp_b), fs)

    # Pick better
    if abs(lag_a) <= abs(lag_b):
        prelim_shift = shift_a
        prelim_lag, prelim_ms = lag_a, ms_a
        chosen_tmp = tmp_a
        other_tmp = tmp_b
    else:
        prelim_shift = shift_b
        prelim_lag, prelim_ms = lag_b, ms_b
        chosen_tmp = tmp_b
        other_tmp = tmp_a

    # Cap applied shift to <= max_ms
    max_samples = int(round(max_ms * fs / 1000.0))
    applied_shift = int(np.sign(prelim_shift) * min(abs(prelim_shift), max_samples))

    # Recompute final by applying capped shift to original GRF
    final_out = out_path
    write_shifted(grf_in, str(final_out), applied_shift)
    final_lag, final_ms, _ = compute_lag_samples(imu_csv, str(final_out), fs)

    # Cleanup temps
    for p in (chosen_tmp, other_tmp):
        try:
            Path(p).unlink(missing_ok=True)
        except Exception:
            pass

    summary = {
        "fs_hz": fs,
        "original": {"lag_samples": int(lag0), "lag_ms": float(ms0)},
        "preliminary": {"shift_samples": int(prelim_shift), "lag_samples": int(prelim_lag), "lag_ms": float(prelim_ms)},
        "applied": {"shift_samples": int(applied_shift)},
        "final": {"lag_samples": int(final_lag), "lag_ms": float(final_ms)},
        "n_overlap": int(n),
    }
    return summary, final_out

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--imu", required=True)
    ap.add_argument("--grf_in", required=True)
    ap.add_argument("--grf_out", required=True)
    ap.add_argument("--fs", type=float, default=0.0, help="Sampling rate Hz; if 0, derived from IMU time_s")
    ap.add_argument("--summary_json", default=None, help="Optional path to write lag summary JSON")
    ap.add_argument("--log_file", default=None, help="Optional log file to tee output; otherwise logs to LOGS_ROOT")
    args = ap.parse_args()

    # Setup logging
    stem = Path(args.grf_in).stem if args.grf_in else "align"
    _setup_logging(f"align_{stem}", args.log_file)

    # Ensure output filename ends with _shifted.csv
    out_path = Path(args.grf_out)
    if out_path.suffix.lower() != ".csv" or "_shifted" not in out_path.stem:
        out_path = out_path.with_name(out_path.stem + "_shifted").with_suffix(".csv")

    # derive fs if needed
    fs_use = args.fs
    if not fs_use or fs_use <= 0:
        try:
            imu_df = pd.read_csv(args.imu)
            if 'time_s' in imu_df.columns:
                t = pd.to_numeric(imu_df['time_s'], errors='coerce').dropna().values
                if t.size > 5:
                    dt = np.median(np.diff(t[:500]))
                    if dt and dt > 0:
                        fs_use = float(round(1.0/dt))
        except Exception:
            fs_use = 200.0
    if not fs_use or fs_use <= 0:
        fs_use = 200.0
    summary, final_out = choose_shift_and_apply(args.imu, args.grf_in, fs_use, out_path, max_ms=10.0)

    print(f"[Before] Best lag: {summary['original']['lag_samples']} samples (~{summary['original']['lag_ms']:.1f} ms)")
    print(f"[Applied] Shift {summary['applied']['shift_samples']:+d} samples")
    print(f"[After]  Best lag: {summary['final']['lag_samples']} samples (~{summary['final']['lag_ms']:.1f} ms)")

    # Write JSON summary
    summary_path = Path(args.summary_json) if args.summary_json else Path(str(final_out) + ".lag.json")
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[OK] wrote: {final_out}")
        print(f"[OK] lag summary: {summary_path}")
    except Exception as e:
        print(f"[WARN] Failed to write summary JSON: {e}")

if __name__ == "__main__":
    main()
