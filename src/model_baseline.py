"""
model_baseline.py

This module provides functionality for feature extraction, training, and evaluation of a baseline model
that predicts GRF from IMU data using ridge regression.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os
import argparse
from dataio import load_imu_csv, load_c3d_grf, resample_grf_to_imu_time


def extract_features(imu_df: pd.DataFrame, window_ms: int = 200, fs_hint: float = 100.0) -> (pd.DataFrame, pd.Series):
    """
    Extract features from IMU data using a sliding window approach.

    Args:
        imu_df (pd.DataFrame): DataFrame containing IMU data.
        window_ms (int): Window length in milliseconds.
        fs_hint (float): Sampling frequency hint.

    Returns:
        pd.DataFrame: Feature DataFrame.
        pd.Series: Time series aligned with imu_df['time_s'].
    """
    acc_mag = np.sqrt(imu_df['ax']**2 + imu_df['ay']**2 + imu_df['az']**2)
    features = []
    times = []
    window_len = max(round(window_ms / 1000 * fs_hint), 3)
    half_window = window_len // 2

    for i in range(half_window, len(imu_df) - half_window):
        window = imu_df.iloc[i - half_window:i + half_window + 1]
        acc_window = acc_mag.iloc[i - half_window:i + half_window + 1]
        feature = {
            'ax_mean': window['ax'].mean(),
            'ay_mean': window['ay'].mean(),
            'az_mean': window['az'].mean(),
            'gx_mean': window['gx'].mean(),
            'gy_mean': window['gy'].mean(),
            'gz_mean': window['gz'].mean(),
            'acc_mag_mean': acc_window.mean(),
            'ax_std': window['ax'].std(),
            'ay_std': window['ay'].std(),
            'az_std': window['az'].std(),
            'gx_std': window['gx'].std(),
            'gy_std': window['gy'].std(),
            'gz_std': window['gz'].std(),
            'acc_mag_std': acc_window.std(),
            'ax_rms': np.sqrt((window['ax']**2).mean()),
            'ay_rms': np.sqrt((window['ay']**2).mean()),
            'az_rms': np.sqrt((window['az']**2).mean()),
            'gx_rms': np.sqrt((window['gx']**2).mean()),
            'gy_rms': np.sqrt((window['gy']**2).mean()),
            'gz_rms': np.sqrt((window['gz']**2).mean()),
            'acc_mag_rms': np.sqrt((acc_window**2).mean())
        }
        features.append(feature)
        times.append(imu_df['time_s'].iloc[i])

    X = pd.DataFrame(features)
    t = pd.Series(times)
    return X, t


def train_and_eval(imu_csv: str, grf_path: str, grf_type: str, bw_kg: float, outdir: str, test_split: float = 0.3):
    """
    Train and evaluate a ridge regression model to predict GRF from IMU data.

    Args:
        imu_csv (str): Path to the IMU CSV file.
        grf_path (str): Path to the GRF file (C3D or CSV).
        grf_type (str): Type of GRF file ('c3d' or 'csv').
        bw_kg (float): Body weight in kilograms.
        outdir (str): Output directory for saving results.
        test_split (float): Fraction of data to use for testing.
    """
    imu_df = load_imu_csv(imu_csv)
    grf_df = load_c3d_grf(grf_path) if grf_type == 'c3d' else load_grf_csv(grf_path)
    grf_df = resample_grf_to_imu_time(imu_df, grf_df)

    X, t = extract_features(imu_df)
    y = grf_df['Fz_N'] / (bw_kg * 9.81)

    split_idx = int(len(X) * (1 - test_split))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    t_test = t[split_idx:]

    model = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=1.0))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * 100
    r = np.corrcoef(y_test, y_pred)[0, 1]

    os.makedirs(outdir, exist_ok=True)
    joblib.dump(model, os.path.join(outdir, 'model.pkl'))

    plt.figure(figsize=(12, 6))
    plt.plot(t_test, y_test, label='True')
    plt.plot(t_test, y_pred, label='Predicted')
    plt.title('Predicted vs True GRF')
    plt.xlabel('Time (s)')
    plt.ylabel('Fz (BW)')
    plt.legend()
    plt.savefig(os.path.join(outdir, 'baseline_pred_vs_truth.png'))

    plt.figure(figsize=(12, 6))
    plt.plot(t_test, y_test - y_pred)
    plt.title('Residuals')
    plt.xlabel('Time (s)')
    plt.ylabel('Residual (BW)')
    plt.savefig(os.path.join(outdir, 'residuals.png'))

    with open(os.path.join(outdir, 'metrics.txt'), 'w') as f:
        f.write(f'RMSE: {rmse:.2f}% BW\n')
        f.write(f'Pearson r: {r:.2f}\n')

    pd.DataFrame({'time_s': t_test, 'Fz_pred_BW': y_pred}).to_csv(os.path.join(outdir, 'predicted_fz.csv'), index=False)

    print(f'Final RMSE: {rmse:.2f}% BW, Pearson r: {r:.2f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate GRF prediction model.')
    parser.add_argument('--imu_csv', required=True, help='Path to IMU CSV file')
    parser.add_argument('--grf_path', required=True, help='Path to GRF file')
    parser.add_argument('--grf_type', choices=['c3d', 'csv'], default='csv', help='Type of GRF file')
    parser.add_argument('--bw_kg', type=float, required=True, help='Body weight in kg')
    parser.add_argument('--outdir', default='out', help='Output directory')
    parser.add_argument('--test_split', type=float, default=0.3, help='Test split fraction')

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    train_and_eval(args.imu_csv, args.grf_path, args.grf_type, args.bw_kg, args.outdir, args.test_split)
