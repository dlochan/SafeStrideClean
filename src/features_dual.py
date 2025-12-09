# src/features_dual.py
import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.signal import welch


def _safe_mean(x: np.ndarray) -> float:
    return float(np.nanmean(x)) if x.size else 0.0


def _safe_std(x: np.ndarray) -> float:
    if x.size <= 1:
        return 0.0
    return float(np.nanstd(x, ddof=0))


def _safe_rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.nanmean(x**2)))


def _safe_ptp(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.nanmax(x) - np.nanmin(x))


def _band_powers(x: np.ndarray, fs: float, bands: List[Tuple[float, float]]) -> List[float]:
    n = x.size
    if n < 4:
        return [0.0] * len(bands)
    nperseg = max(8, min(n, 128))
    try:
        f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=int(0.5 * nperseg))
        if f.size == 0:
            return [0.0] * len(bands)
        df = np.mean(np.diff(f)) if f.size > 1 else 1.0
        out = []
        for lo, hi in bands:
            m = (f >= lo) & (f < hi)
            out.append(float(np.nansum(Pxx[m]) * df) if np.any(m) else 0.0)
        return out
    except Exception:
        return [0.0] * len(bands)


def _conv_bank_reduce(x: np.ndarray, kernels: List[List[float]], reduce: str = "mean_abs") -> List[float]:
    if x.size == 0:
        return [0.0] * len(kernels)
    feats = []
    for k in kernels:
        karr = np.asarray(k, dtype=float)
        if karr.size == 0:
            feats.append(0.0)
            continue
        y = np.convolve(x, karr, mode="valid") if x.size >= karr.size else np.array([])
        if y.size == 0:
            feats.append(0.0)
        elif reduce == "mean_abs":
            feats.append(float(np.nanmean(np.abs(y))))
        else:
            feats.append(float(np.nanmean(y)))
    return feats


def _get_tags(imu_cols: List[str]) -> Dict[str, Dict[str, str]]:
    base = ["ax", "ay", "az", "gx", "gy", "gz"]
    cols = set(imu_cols)
    # Prefer explicit thigh/shank schema
    thigh_ok = all(f"{c}_thigh" in cols for c in base)
    shank_ok = all(f"{c}_shank" in cols for c in base)
    if thigh_ok and shank_ok:
        return {
            "thigh": {k: f"{k}_thigh" for k in base},
            "shank": {k: f"{k}_shank" for k in base},
        }
    # Fallback to suffix tags
    tags: Dict[str, Dict[str, str]] = {}
    for c in imu_cols:
        for k in base:
            pref = f"{k}_"
            if c.startswith(pref):
                tag = c[len(pref):]
                tags.setdefault(tag, {})[k] = c
    # Keep only complete
    return {t: m for t, m in tags.items() if len(m) == 6}


def window_centers(n: int, fs: float, window_ms: int) -> Tuple[np.ndarray, np.ndarray]:
    win = int(round(fs * window_ms / 1000.0))
    win = max(3, win)
    half = win // 2
    idx = np.arange(half, n - half)
    return idx, half


def build_dual_features(
    imu: pd.DataFrame,
    fs: float = 200.0,
    window_ms: int = 200,
    bands: Optional[List[Tuple[float, float]]] = None,
    conv_kernels: Optional[List[List[float]]] = None,
    conv_reduce: str = "mean_abs",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Compute windowed features for dual sensors (thigh + shank) from IMU dataframe.
    Expects columns for both sensors, e.g., ax_thigh ... gz_thigh and ax_shank ... gz_shank.
    Returns X (n_windows x n_features) and time centers (Series name='time_s').
    """
    if "time_s" not in imu.columns:
        raise ValueError("IMU must include 'time_s'")

    tags = _get_tags(list(imu.columns))
    if not ("thigh" in tags and "shank" in tags):
        raise ValueError(f"Expected 'thigh' and 'shank' tags; got {list(tags.keys())}")

    if bands is None:
        bands = [(0.5, 3.0), (3.0, 8.0), (8.0, 15.0), (15.0, 30.0)]
    if conv_kernels is None:
        conv_kernels = [[1, -1], [1, 0, -1], [1, 2, 1], [-1, 2, -1]]

    idx, half = window_centers(len(imu), fs, window_ms)

    times = []
    rows: List[List[float]] = []

    for i in idx:
        w = imu.iloc[i - half : i + half]
        feats: List[float] = []
        per_tag = {}
        for tag in ("thigh", "shank"):
            m = tags[tag]
            # Build arrays
            ax = w[m["ax"]].to_numpy(float)
            ay = w[m["ay"]].to_numpy(float)
            az = w[m["az"]].to_numpy(float)
            gx = w[m["gx"]].to_numpy(float)
            gy = w[m["gy"]].to_numpy(float)
            gz = w[m["gz"]].to_numpy(float)

            # Basic stats and RMS
            means = [_safe_mean(ax), _safe_mean(ay), _safe_mean(az), _safe_mean(gx), _safe_mean(gy), _safe_mean(gz)]
            stds  = [_safe_std(ax),  _safe_std(ay),  _safe_std(az),  _safe_std(gx),  _safe_std(gy),  _safe_std(gz)]
            rms   = [_safe_rms(ax),  _safe_rms(ay),  _safe_rms(az),  _safe_rms(gx),  _safe_rms(gy),  _safe_rms(gz)]

            # Magnitudes
            acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
            gyr_mag = np.sqrt(gx**2 + gy**2 + gz**2)
            acc_mag_mean = _safe_mean(acc_mag)
            acc_mag_std  = _safe_std(acc_mag)
            acc_mag_rms  = _safe_rms(acc_mag)
            gyr_mag_mean = _safe_mean(gyr_mag)
            gyr_mag_std  = _safe_std(gyr_mag)
            gyr_mag_rms  = _safe_rms(gyr_mag)

            # Deltas (first diff) and jerk (second diff on acc)
            d_acc = np.diff(acc_mag)
            jerk = np.diff(np.diff(acc_mag)) * fs  # approximate derivative change
            d_acc_mean = _safe_mean(d_acc) if d_acc.size else 0.0
            d_acc_std  = _safe_std(d_acc)  if d_acc.size else 0.0
            jerk_rms   = _safe_rms(jerk)   if jerk.size else 0.0

            # Peak-to-peak per axis (acc)
            ptp_ax = _safe_ptp(ax)
            ptp_ay = _safe_ptp(ay)
            ptp_az = _safe_ptp(az)

            # Welch bands on acc magnitude
            band_pw = _band_powers(acc_mag, fs, bands)

            # Conv bank features on acc magnitude
            conv_feats = _conv_bank_reduce(acc_mag, conv_kernels)

            per_tag[tag] = {
                "means": means,
                "stds": stds,
                "rms": rms,
                "acc_mag": (acc_mag_mean, acc_mag_std, acc_mag_rms),
                "gyr_mag": (gyr_mag_mean, gyr_mag_std, gyr_mag_rms),
                "deltas": (d_acc_mean, d_acc_std),
                "jerk": (jerk_rms,),
                "ptp": (ptp_ax, ptp_ay, ptp_az),
                "welch": band_pw,
                "conv": conv_feats,
                "acc_mag_series": acc_mag,
            }

        # Cross-sensor correlation between acc magnitudes (small lag window)
        a = per_tag["thigh"]["acc_mag_series"]
        b = per_tag["shank"]["acc_mag_series"]
        # normalize
        a0 = a - np.nanmean(a)
        b0 = b - np.nanmean(b)
        denom = np.nanstd(a0) * np.nanstd(b0)
        corr0 = 0.0 if denom == 0 or np.isnan(denom) else float(np.nanmean(a0 * b0) / denom)
        # scan lags +/-5 samples
        max_corr = corr0
        max_lag = 0
        for lag in range(-5, 6):
            if lag == 0:
                continue
            if lag > 0:
                a1 = a0[lag:]
                b1 = b0[:-lag]
            else:
                k = -lag
                a1 = a0[:-k]
                b1 = b0[k:]
            if a1.size == 0 or b1.size == 0:
                continue
            d = np.nanstd(a1) * np.nanstd(b1)
            if d == 0 or np.isnan(d):
                continue
            c = float(np.nanmean(a1 * b1) / d)
            if abs(c) > abs(max_corr):
                max_corr, max_lag = c, lag

        # Pack features in fixed order per tag
        def pack_tag(tag: str) -> List[float]:
            t = per_tag[tag]
            return (
                t["means"] + t["stds"] + t["rms"] +
                list(t["acc_mag"]) + list(t["gyr_mag"]) +
                list(t["deltas"]) + list(t["jerk"]) + list(t["ptp"]) +
                list(t["welch"]) + list(t["conv"]) 
            )

        feats.extend(pack_tag("thigh"))
        feats.extend(pack_tag("shank"))
        feats.extend([corr0, max_corr, float(max_lag)])

        rows.append(feats)
        times.append(float(imu["time_s"].iloc[i]))

    # Columns
    def names_for(tag: str) -> List[str]:
        ch = ["ax","ay","az","gx","gy","gz"]
        cols = (
            [f"mean_{c}_{tag}" for c in ch] +
            [f"std_{c}_{tag}"  for c in ch] +
            [f"rms_{c}_{tag}"  for c in ch] +
            [f"acc_mag_mean_{tag}", f"acc_mag_std_{tag}", f"acc_mag_rms_{tag}"] +
            [f"gyr_mag_mean_{tag}", f"gyr_mag_std_{tag}", f"gyr_mag_rms_{tag}"] +
            [f"acc_mag_diff_mean_{tag}", f"acc_mag_diff_std_{tag}"] +
            [f"jerk_rms_{tag}"] +
            [f"ptp_ax_{tag}", f"ptp_ay_{tag}", f"ptp_az_{tag}"] +
            [f"welch_band{i+1}_{tag}" for i in range(len(bands))] +
            [f"conv{i+1}_{tag}" for i in range(len(conv_kernels))]
        )
        return cols

    columns = names_for("thigh") + names_for("shank") + ["acc_mag_corr0", "acc_mag_corr_max", "acc_mag_corr_max_lag"]

    X = pd.DataFrame(rows, columns=columns)
    t = pd.Series(times, name="time_s")
    # Ensure no NaNs
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return X, t
