import pandas as pd, numpy as np

IMU = r"data\working\gt_AB01_cutting_leftfast_imu.csv"
GRF = r"data\working\gt_AB01_cutting_leftfast_grf_shifted.csv"
FS = 200.0

imu = pd.read_csv(IMU)
grf = pd.read_csv(GRF)

acc_mag = np.sqrt(imu.ax**2 + imu.ay**2 + imu.az**2)
fz = grf.Fz_N.fillna(0.0)

# detrend-light
acc_mag = acc_mag - acc_mag.mean()
fz = fz - fz.mean()

n = min(len(acc_mag), len(fz))
acc_mag = acc_mag.iloc[:n].to_numpy()
fz = fz.iloc[:n].to_numpy()

xc = np.correlate(acc_mag, fz, mode="full")
lags = np.arange(-n+1, n)
best = int(lags[np.argmax(xc)])
print(f"Best lag after sample-shift: {best} samples (~{1000*best/FS:.1f} ms)")
