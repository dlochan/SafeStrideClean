"""
filters.py

This module provides filtering functions for IMU and GRF data using Butterworth filters.
"""

import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt


def bandpass_imu(df, fs=100.0, low=0.5, high=20.0):
    """
    2nd-order Butterworth bandpass filter on ax, ay, az, gx, gy, gz.
    Returns a copy of the DataFrame with filtered data.

    Args:
        df (pd.DataFrame): DataFrame containing IMU data.
        fs (float): Sampling frequency in Hz.
        low (float): Low cutoff frequency in Hz.
        high (float): High cutoff frequency in Hz.

    Returns:
        pd.DataFrame: DataFrame with bandpass-filtered IMU data.
    """
    cols = ["ax", "ay", "az", "gx", "gy", "gz"]
    out = df.copy()
    nyq = 0.5 * fs
    b, a = butter(2, [low/nyq, high/nyq], btype="bandpass")
    for c in cols:
        out[c] = filtfilt(b, a, out[c].values)
    return out


def lowpass_grf(df, fs=1000.0, cutoff=20.0):
    """
    2nd-order Butterworth lowpass filter on Fx_N, Fy_N, Fz_N.
    Returns a copy of the DataFrame with filtered data.

    Args:
        df (pd.DataFrame): DataFrame containing GRF data.
        fs (float): Sampling frequency in Hz.
        cutoff (float): Cutoff frequency in Hz.

    Returns:
        pd.DataFrame: DataFrame with lowpass-filtered GRF data.
    """
    cols = ["Fx_N", "Fy_N", "Fz_N"]
    out = df.copy()
    nyq = 0.5 * fs
    b, a = butter(2, cutoff/nyq, btype="low")
    for c in cols:
        out[c] = filtfilt(b, a, out[c].values)
    return out
