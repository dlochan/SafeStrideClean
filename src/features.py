import pandas as pd
import numpy as np
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler


def window_index(n_samples: int, fs: float, window_ms: int) -> (int, int):
    """
    Calculate window indices for feature extraction.

    Returns:
      (half window size in samples, full window length in samples)
    """
    win_len = int(round(fs * window_ms / 1000.0))
    # be robust to tiny signals
    win_len = max(win_len, 3)
    half_win = win_len // 2
    return half_win, win_len


def _safe_std(x: np.ndarray) -> float:
    if x.size <= 1:
        return 0.0
    return float(np.std(x, ddof=1))


def _safe_rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x**2)))


def _safe_ptp(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.max(x) - np.min(x))


def _welch_band_power(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    """
    Band-limited power via Welch, robust to very short windows.
    Returns 0.0 if too few points to estimate.
    """
    n = x.size
    if n < 4:
        return 0.0
    # choose a small, valid nperseg
    nperseg = max(8, min(n, 64))
    try:
        f, Pxx = welch(x, fs=fs, nperseg=nperseg, noverlap=int(0.5 * nperseg))
        if f.size == 0 or Pxx.size == 0:
            return 0.0
        mask = (f >= lo) & (f < hi)
        if not np.any(mask):
            return 0.0
        # integrate: sum(Pxx * df)
        if f.size > 1:
            df = np.mean(np.diff(f))
            return float(np.sum(Pxx[mask]) * df)
        else:
            return float(np.sum(Pxx[mask]))
    except Exception:
        return 0.0


# in src/features.py — keep your imports; replace rolling_features
import pandas as pd
import numpy as np
from scipy.signal import welch

def _find_sensor_sets(imu_cols):
    """
    Return a dict: {sensor_tag: {'ax':col,...,'gz':col}}.
    Two patterns supported:
      1) single-sensor: exact names ax,ay,az,gx,gy,gz  -> tag ''
      2) multi-sensor:  ax_<tag>,...,gz_<tag>          -> tag '<tag>'
    """
    base = ["ax","ay","az","gx","gy","gz"]
    cols = set(imu_cols)

    # single-sensor case
    if all(c in cols for c in base):
        return {"": {k: k for k in base}}

    # multi-sensor case: collect suffix tags
    tags = {}
    for c in imu_cols:
        for k in base:
            prefix = f"{k}_"
            if c.startswith(prefix):
                tag = c[len(prefix):]
                tags.setdefault(tag, {})[k] = c

    # keep only complete sets
    complete = {tag: mapping for tag, mapping in tags.items() if len(mapping) == 6}
    if not complete:
        raise ValueError(f"No complete IMU 6-axis sets found. Columns: {imu_cols}")
    return complete

def rolling_features(imu: pd.DataFrame, fs: float, window_ms: int = 200) -> (pd.DataFrame, pd.Series):
    """
    If single-sensor: expects ['time_s','ax','ay','az','gx','gy','gz'].
    If multi-sensor: expects ['time_s', 'ax_<tag>',...,'gz_<tag>', ...] for 2+ tags.
    Returns:
      X: features concatenated for each sensor tag (columns include suffixes when multi-sensor)
      t: time_s aligned to window centers
    """
    if "time_s" not in imu.columns:
        raise ValueError("IMU dataframe must include 'time_s'.")

    # discover sensor sets
    sets = _find_sensor_sets(list(imu.columns))
    base = ["ax","ay","az","gx","gy","gz"]

    # windowing
    win_len = max(1, int(fs * window_ms / 1000))
    half_win = max(1, win_len // 2)

    times = []
    feats_all = []

    # iterate windows
    for i in range(half_win, len(imu) - half_win):
        row_feats = []

        # per-sensor features
        for tag, mapping in sets.items():
            # slice window for this sensor
            w = imu.iloc[i-half_win:i+half_win]
            # build ndarray for convenience
            M = np.vstack([w[mapping[k]].to_numpy(dtype=float) for k in base])  # shape (6, win)

            # means/stds/rms for the six channels
            means = M.mean(axis=1)
            stds  = M.std(axis=1, ddof=0)
            rms   = np.sqrt((M**2).mean(axis=1))

            # acceleration magnitude features from (ax,ay,az)
            acc_mag = np.sqrt((M[0]**2) + (M[1]**2) + (M[2]**2))
            acc_mag_mean = acc_mag.mean()
            acc_mag_std  = acc_mag.std()
            acc_mag_rms  = np.sqrt((acc_mag**2).mean())
            acc_mag_diff = np.diff(acc_mag)
            acc_mag_diff_mean = acc_mag_diff.mean() if len(acc_mag_diff) else 0.0
            acc_mag_diff_std  = acc_mag_diff.std()  if len(acc_mag_diff) else 0.0

            # ptp for linear acc only
            ptp_ax = w[mapping["ax"]].max() - w[mapping["ax"]].min()
            ptp_ay = w[mapping["ay"]].max() - w[mapping["ay"]].min()
            ptp_az = w[mapping["az"]].max() - w[mapping["az"]].min()

            # spectral bands from acc_mag (protect nperseg)
            nperseg = max(8, min(len(acc_mag), half_win))
            f, Pxx = welch(acc_mag, fs=fs, nperseg=nperseg)
            band_1_power = Pxx[(f >= 0.5) & (f < 3)].sum() if len(Pxx) else 0.0
            band_2_power = Pxx[(f >= 3)   & (f < 12)].sum() if len(Pxx) else 0.0

            # pack with tag suffix when multi-sensor
            suf = "" if tag == "" else f"_{tag}"
            row_feats.extend(list(means) + list(stds) + list(rms) + [
                acc_mag_mean, acc_mag_std, acc_mag_rms,
                acc_mag_diff_mean, acc_mag_diff_std,
                ptp_ax, ptp_ay, ptp_az, band_1_power, band_2_power
            ])

        feats_all.append(row_feats)
        times.append(imu["time_s"].iloc[i])

    # build column names
    def names_for_tag(tag):
        suf = "" if tag == "" else f"_{tag}"
        ch = ["ax","ay","az","gx","gy","gz"]
        cols = (
            [f"mean_{c}{suf}" for c in ch] +
            [f"std_{c}{suf}"  for c in ch] +
            [f"rms_{c}{suf}"  for c in ch] +
            [f"acc_mag_mean{suf}", f"acc_mag_std{suf}", f"acc_mag_rms{suf}",
             f"acc_mag_diff_mean{suf}", f"acc_mag_diff_std{suf}",
             f"ptp_ax{suf}", f"ptp_ay{suf}", f"ptp_az{suf}",
             f"band_1_power{suf}", f"band_2_power{suf}"]
        )
        return cols

    colnames = []
    for tag in sets.keys():
        colnames += names_for_tag(tag)

    X = pd.DataFrame(feats_all, columns=colnames)
    t = pd.Series(times, name="time_s")
    return X, t


def normalize_features(X: pd.DataFrame) -> (np.ndarray, StandardScaler):
    """
    Normalize features using StandardScaler.

    Returns:
      (Scaled features as np.ndarray, fitted StandardScaler)
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
