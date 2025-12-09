# scripts/check_alignment.py
import pandas as pd, numpy as np
from pathlib import Path

IMU = r"data\working\gt_AB01_cutting_leftfast_imu.csv"
GRF = r"data\working\gt_AB01_cutting_leftfast_grf.csv"
FS = 200.0  # Hz

imu = pd.read_csv(IMU)
grf = pd.read_csv(GRF)

# basic sanity
assert "time_s" in imu and "time_s" in grf
print("IMU time [s]:", float(imu.time_s.min()), "→", float(imu.time_s.max()), "n=", len(imu))
print("GRF time [s]:", float(grf.time_s.min()), "→", float(grf.time_s.max()), "n=", len(grf))

# build simple signals to correlate
acc_mag = np.sqrt(imu.ax**2 + imu.ay**2 + imu.az**2)
fz = grf.Fz_N.fillna(0.0)

# detrend lightly
acc_mag = acc_mag - acc_mag.mean()
fz = fz - fz.mean()

# same length for correlation
n = min(len(acc_mag), len(fz))
acc_mag = acc_mag.iloc[:n].to_numpy()
fz = fz.iloc[:n].to_numpy()

# cross-correlation to estimate lag (samples)
xc = np.correlate(acc_mag, fz, mode="full")
lags = np.arange(-n+1, n)
best = int(lags[np.argmax(xc)])
lag_ms = 1000.0 * best / FS

print(f"Estimated lag (IMU leads + / lags -): {best} samples ≈ {lag_ms:.1f} ms")
print("Rule of thumb: |lag| < 20 ms is typically fine with 200 Hz data.")
