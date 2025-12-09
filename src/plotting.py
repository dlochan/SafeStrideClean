"""
plotting.py

This module provides functionality to load IMU and GRF data, resample GRF to IMU time, and plot the results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataio import load_imu_csv, load_c3d_grf, resample_grf_to_imu_time


def plot_imu_and_grf(imu_path, grf_path, out_png, grf_type="c3d"):
    """
    Loads IMU CSV and GRF (C3D or CSV), resamples GRF to IMU time,
    then plots accel magnitude (top) and vertical GRF Fz (bottom).
    Saves to out_png.
    """
    imu = load_imu_csv(imu_path)
    if grf_type == "c3d":
        grf = load_c3d_grf(grf_path)
    else:
        grf = load_grf_csv(grf_path)

    grf_i = resample_grf_to_imu_time(imu, grf)
    acc_mag = np.sqrt(imu["ax"]**2 + imu["ay"]**2 + imu["az"]**2)

    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(imu["time_s"], acc_mag)
    plt.title("IMU acceleration magnitude")
    plt.xlabel("Time (s)")
    plt.ylabel("m/s^2")

    plt.subplot(2,1,2)
    plt.plot(grf_i["time_s"], grf_i["Fz_N"])
    plt.title("Vertical GRF (Fz)")
    plt.xlabel("Time (s)")
    plt.ylabel("N")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"[OK] Saved plot to {out_png}")


if __name__ == "__main__":
    import os
    imu_path = os.path.join("data","sample","imu_trial01.csv")
    # Try C3D first; if missing, fall back to CSV
    c3d_path = os.path.join("data","sample","grf_trial01.c3d")
    csv_path = os.path.join("data","sample","grf_trial01.csv")
    out_png = "sample_plot.png"

    if os.path.exists(imu_path) and os.path.exists(c3d_path):
        plot_imu_and_grf(imu_path, c3d_path, out_png, grf_type="c3d")
    elif os.path.exists(imu_path) and os.path.exists(csv_path):
        plot_imu_and_grf(imu_path, csv_path, out_png, grf_type="csv")
    else:
        print("[INFO] Put imu_trial01.csv and a GRF file in data/sample/ and re-run.")
