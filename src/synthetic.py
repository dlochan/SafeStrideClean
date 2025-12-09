import numpy as np
import pandas as pd
import os


def make_synthetic_imu_grf(duration_s=10.0, fs=100.0, seed=42) -> (pd.DataFrame, pd.DataFrame):
    """
    Generate synthetic IMU and GRF data.

    Parameters:
    - duration_s: float : Duration of the synthetic data in seconds.
    - fs: float : Sampling frequency in Hz.
    - seed: int : Random seed for reproducibility.

    Returns:
    - tuple : (IMU DataFrame, GRF DataFrame)
    """
    np.random.seed(seed)
    n_samples = int(duration_s * fs)
    time_s = np.arange(n_samples) / fs

    # Generate synthetic IMU data
    ax = np.sin(2 * np.pi * 0.5 * time_s) + 0.1 * np.random.randn(n_samples)
    ay = np.sin(2 * np.pi * 0.5 * time_s + np.pi / 4) + 0.1 * np.random.randn(n_samples)
    az = 9.81 + 0.1 * np.random.randn(n_samples)
    gx = 0.01 * np.random.randn(n_samples)
    gy = 0.01 * np.random.randn(n_samples)
    gz = 0.01 * np.random.randn(n_samples)

    imu_df = pd.DataFrame({'time_s': time_s, 'ax': ax, 'ay': ay, 'az': az, 'gx': gx, 'gy': gy, 'gz': gz})

    # Generate synthetic GRF data
    Fz_N = np.zeros(n_samples)
    peak_indices = np.random.choice(n_samples, size=2, replace=False)
    Fz_N[peak_indices] = 1000  # Two peaks

    grf_df = pd.DataFrame({'time_s': time_s, 'Fx_N': np.zeros(n_samples), 'Fy_N': np.zeros(n_samples), 'Fz_N': Fz_N})

    return imu_df, grf_df


def write_sample_files(out_dir="data/sample"):
    """
    Write synthetic IMU and GRF data to CSV files.

    Parameters:
    - out_dir: str : Directory to save the sample files.
    """
    os.makedirs(out_dir, exist_ok=True)
    imu_df, grf_df = make_synthetic_imu_grf()
    imu_path = os.path.join(out_dir, 'imu_trial01.csv')
    grf_path = os.path.join(out_dir, 'grf_trial01.csv')
    imu_df.to_csv(imu_path, index=False)
    grf_df.to_csv(grf_path, index=False)
    print(f"IMU data written to {imu_path}")
    print(f"GRF data written to {grf_path}")


if __name__ == "__main__":
    write_sample_files()
