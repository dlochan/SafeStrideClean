# scripts/shift_grf_by_samples.py
# Shift the GRF *values* by a fixed number of samples.
# Positive SHIFT_SAMPLES delays the GRF (pads zeros at the start).
# Negative SHIFT_SAMPLES advances it (cuts off the start).

import pandas as pd
import numpy as np
from pathlib import Path

IN  = r"data\working\gt_AB01_cutting_leftfast_grf.csv"           # current (unshifted) total GRF
OUT = r"data\working\gt_AB01_cutting_leftfast_grf_shifted.csv"   # will overwrite/create
SHIFT_SAMPLES = 25  # +25 samples = +125 ms at 200 Hz

df = pd.read_csv(IN)
cols_force = [c for c in df.columns if c.lower() in ("fx_n","fy_n","fz_n")]

if len(cols_force) != 3 or "time_s" not in df.columns:
    raise SystemExit(f"Unexpected columns in {IN}. Got: {list(df.columns)}")

n = len(df)
shift = SHIFT_SAMPLES

def shift_series(s: pd.Series, k: int) -> pd.Series:
    if k == 0:
        return s
    if k > 0:
        # delay: pad front with zeros, drop tail
        return pd.Series(np.r_[np.zeros(k), s.values[:-k]], index=s.index)
    else:
        k = -k
        # advance: drop head, pad zeros at end
        return pd.Series(np.r_[s.values[k:], np.zeros(k)], index=s.index)

df_shifted = df.copy()
for c in cols_force:
    df_shifted[c] = shift_series(df[c].fillna(0.0), shift)

df_shifted.to_csv(OUT, index=False)
print("[OK] wrote:", Path(OUT).resolve())
print("Forces shifted by", SHIFT_SAMPLES, "samples.")
print(df_shifted.head(3).to_string(index=False))
