from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from .windowing import WindowedIMU


@dataclass(frozen=True)
class FeatureTensor:
    X: np.ndarray  # (N, C_out, L)
    channel_names: List[str]


def extract_features(
    windowed: WindowedIMU,
    *,
    include_magnitude: bool = False,
) -> FeatureTensor:
    """Convert windowed IMU signals into a model-ready tensor.

    Output is shaped (num_windows, num_channels, window_len).

    The default feature set is minimal and deterministic:
    - raw signals
    - per-channel mean/std/rms broadcast across the window length

    If include_magnitude=True, adds accel/gyro magnitudes per sensor (if channel
    names follow the ax_<sensor>/... convention).
    """

    X = np.asarray(windowed.windows, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"windows must have shape (N,C,L), got {X.shape}")
    N, C, L = X.shape
    if N <= 0 or C <= 0 or L <= 0:
        raise ValueError(f"windows must be non-empty, got {X.shape}")

    feats = [X]
    feat_names = list(windowed.channel_names)

    mean = X.mean(axis=2, keepdims=True)
    std = X.std(axis=2, keepdims=True)
    rms = np.sqrt((X * X).mean(axis=2, keepdims=True))

    feats.append(np.repeat(mean, L, axis=2))
    feat_names.extend([f"mean_{c}" for c in windowed.channel_names])

    feats.append(np.repeat(std, L, axis=2))
    feat_names.extend([f"std_{c}" for c in windowed.channel_names])

    feats.append(np.repeat(rms, L, axis=2))
    feat_names.extend([f"rms_{c}" for c in windowed.channel_names])

    if include_magnitude:
        mag_ts, mag_names = _magnitude_channels(X, windowed.channel_names)
        if mag_ts.size:
            feats.append(mag_ts)
            feat_names.extend(mag_names)

            mag_mean = mag_ts.mean(axis=2, keepdims=True)
            mag_std = mag_ts.std(axis=2, keepdims=True)
            mag_rms = np.sqrt((mag_ts * mag_ts).mean(axis=2, keepdims=True))

            feats.append(np.repeat(mag_mean, L, axis=2))
            feat_names.extend([f"mean_{c}" for c in mag_names])

            feats.append(np.repeat(mag_std, L, axis=2))
            feat_names.extend([f"std_{c}" for c in mag_names])

            feats.append(np.repeat(mag_rms, L, axis=2))
            feat_names.extend([f"rms_{c}" for c in mag_names])

    X_out = np.concatenate(feats, axis=1).astype(np.float32)
    return FeatureTensor(X=X_out, channel_names=feat_names)


def _magnitude_channels(X: np.ndarray, channel_names: Sequence[str]) -> tuple[np.ndarray, List[str]]:
    """Compute per-sensor accel/gyro magnitudes for channels named ax_<s>,ay_<s>,az_<s> etc."""

    # Build indices for each sensor based on suffix.
    idx_by_name = {name: i for i, name in enumerate(channel_names)}

    sensors = []
    for name in channel_names:
        parts = name.split("_", 1)
        if len(parts) != 2:
            continue
        _, sensor = parts
        if sensor and sensor not in sensors:
            sensors.append(sensor)

    mag_series = []
    mag_names: list[str] = []
    for sensor in sensors:
        ax = idx_by_name.get(f"ax_{sensor}")
        ay = idx_by_name.get(f"ay_{sensor}")
        az = idx_by_name.get(f"az_{sensor}")
        gx = idx_by_name.get(f"gx_{sensor}")
        gy = idx_by_name.get(f"gy_{sensor}")
        gz = idx_by_name.get(f"gz_{sensor}")

        if ax is not None and ay is not None and az is not None:
            a = X[:, [ax, ay, az], :]
            amag = np.sqrt((a * a).sum(axis=1, keepdims=True))
            mag_series.append(amag)
            mag_names.append(f"acc_mag_{sensor}")

        if gx is not None and gy is not None and gz is not None:
            g = X[:, [gx, gy, gz], :]
            gmag = np.sqrt((g * g).sum(axis=1, keepdims=True))
            mag_series.append(gmag)
            mag_names.append(f"gyro_mag_{sensor}")

    if not mag_series:
        return np.zeros((X.shape[0], 0, X.shape[2]), dtype=np.float32), []

    return np.concatenate(mag_series, axis=1).astype(np.float32), mag_names
