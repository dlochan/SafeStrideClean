# scripts/force_restandardize_gt_trial.py
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os
from src.adapters.gt_noncyclic import load_gt_imu_real, load_gt_grf_total

IMU_IN = r"E:\safestride\datasets\ProcessedData\AB01\cutting_1_left-fast\AB01_cutting_1_left-fast_imu_real.csv"
GRF_IN = r"E:\safestride\datasets\ProcessedData\AB01\cutting_1_left-fast\AB01_cutting_1_left-fast_grf.csv"

OUT_IMU = r"data\working\gt_AB01_cutting_leftfast_imu.csv"
OUT_GRF = r"data\working\gt_AB01_cutting_leftfast_grf.csv"

def main():
    # Pick ONE sensor for now (change 'rshank' if you prefer another)
    imu = load_gt_imu_real(IMU_IN, sensor="rshank")
    grf = load_gt_grf_total(GRF_IN)

    # Safety check
    want_imu = ['time_s','ax','ay','az','gx','gy','gz']
    want_grf = ['time_s','Fx_N','Fy_N','Fz_N']
    assert list(imu.columns) == want_imu, f"IMU cols {list(imu.columns)} != {want_imu}"
    assert list(grf.columns) == want_grf, f"GRF cols {list(grf.columns)} != {want_grf}"

    os.makedirs("data\\working", exist_ok=True)
    imu.to_csv(OUT_IMU, index=False)
    grf.to_csv(OUT_GRF, index=False)

    print("[OK] wrote standardized:")
    print("  ", Path(OUT_IMU).resolve())
    print("  ", Path(OUT_GRF).resolve())
    print("\nIMU preview:\n", imu.head(3).to_string(index=False))
    print("\nGRF preview:\n", grf.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
