# scripts/shift_grf_time.py
# Shift GRF time forward by +0.125 s to correct a -25 sample lag at 200 Hz.

import pandas as pd
from pathlib import Path

IN  = r"data\working\gt_AB01_cutting_leftfast_grf.csv"
OUT = r"data\working\gt_AB01_cutting_leftfast_grf_shifted.csv"
SHIFT_S = 0.125  # 125 ms

df = pd.read_csv(IN)
if "time_s" not in df.columns:
    raise SystemExit(f"No time_s column in {IN}. Columns: {df.columns.tolist()}")

df["time_s"] = df["time_s"] + SHIFT_S
df.to_csv(OUT, index=False)
print("[OK] wrote:", Path(OUT).resolve())
print(df.head(3).to_string(index=False))
