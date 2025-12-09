# scripts/standardize_gt_trial.py
# Convert one GT trial into the standardized 7-column IMU + 3-axis GRF your pipeline expects.

import os
from src.adapters.gt_noncyclic import (
    load_gt_imu_real,
    load_gt_grf,
    load_gt_activity_flag,
    filter_active,
)

# EDIT THESE PATHS ONLY IF YOURS ARE DIFFERENT
imu_path  = r"E:\safestride\datasets\ProcessedData\AB01\cutting_1_left-fast\AB01_cutting_1_left-fast_imu_real.csv"
grf_path  = r"E:\safestride\datasets\ProcessedData\AB01\cutting_1_left-fast\AB01_cutting_1_left-fast_grf.csv"
flag_path = r"E:\safestride\datasets\ProcessedData\AB01\cutting_1_left-fast\AB01_cutting_1_left-fast_activity_flag.csv"

out_imu = r"data\working\gt_AB01_cutting_leftfast_imu.csv"
out_grf = r"data\working\gt_AB01_cutting_leftfast_grf.csv"

def main():
    imu = load_gt_imu_real(imu_path)
    grf = load_gt_grf(grf_path)

    # Try to filter to active frames if the flag file is present/valid
    try:
        flag = load_gt_activity_flag(flag_path)
        imu, grf = filter_active(imu, grf, flag)
        print(f"[OK] filtered to active: IMU {imu.shape}, GRF {grf.shape}")
    except Exception as e:
        print(f"[WARN] no activity filtering: {e}")

    os.makedirs(os.path.dirname(out_imu), exist_ok=True)
    imu.to_csv(out_imu, index=False)
    grf.to_csv(out_grf, index=False)

    print("[OK] wrote:")
    print("  ", os.path.abspath(out_imu))
    print("  ", os.path.abspath(out_grf))

    print("\nIMU preview:")
    print(imu.head(3).to_string(index=False))
    print("\nGRF preview:")
    print(grf.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
