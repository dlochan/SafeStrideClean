from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def _safe_diff(x: np.ndarray) -> np.ndarray:
    if len(x) < 2:
        return np.zeros_like(x)
    d = np.diff(x, prepend=x[0])
    return d


def basic_time_features(df: pd.DataFrame) -> dict:
    t = df['time_s'].to_numpy(dtype=float) if 'time_s' in df.columns else None
    fzbw = df['Fz_%BW'].to_numpy(dtype=float) if 'Fz_%BW' in df.columns else None
    if fzbw is None:
        # compute from Fz_N if present using median BW over window; but without BW we skip
        return {}
    out = {}
    out['peak_fz_bw'] = float(np.nanmax(fzbw)) if len(fzbw) else np.nan
    out['mean_fz_bw'] = float(np.nanmean(fzbw)) if len(fzbw) else np.nan
    if t is not None and len(t) == len(fzbw) and len(t) > 1:
        dt = np.diff(t).mean()
        out['impulse_bw'] = float(np.nansum(np.maximum(fzbw, 0) * dt))
        # loading rate as max positive slope over a small window
        slope = _safe_diff(fzbw) / np.maximum(_safe_diff(t), 1e-9)
        out['loading_rate_bw'] = float(np.nanmax(slope))
        # simple variability
        out['std_fz_bw'] = float(np.nanstd(fzbw))
    else:
        out['impulse_bw'] = np.nan
        out['loading_rate_bw'] = np.nan
        out['std_fz_bw'] = np.nan
    return out


def basic_freq_features(df: pd.DataFrame) -> dict:
    if 'time_s' not in df.columns or 'Fz_%BW' not in df.columns:
        return {}
    t = df['time_s'].to_numpy(dtype=float)
    y = df['Fz_%BW'].to_numpy(dtype=float)
    if len(t) < 8:
        return {}
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        return {}
    fs = 1.0 / dt
    y = y - np.nanmean(y)
    n = len(y)
    # FFT power spectrum
    Y = np.fft.rfft(y, n=n)
    freqs = np.fft.rfftfreq(n, d=dt)
    P = (np.abs(Y) ** 2) / n
    def band_power(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        if not m.any():
            return 0.0
        return float(P[m].sum())
    return {
        'pwr_0_3hz': band_power(0.0, 3.0),
        'pwr_3_8hz': band_power(3.0, 8.0),
        'pwr_8_15hz': band_power(8.0, 15.0),
    }


def compute_second_layer_features(pred_csv: str | Path) -> dict:
    df = pd.read_csv(pred_csv)
    feats = {}
    feats.update(basic_time_features(df))
    feats.update(basic_freq_features(df))
    return feats
