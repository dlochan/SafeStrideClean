from __future__ import annotations

from pathlib import Path

import numpy as np

from src.imu.schema import parse_imu_csv
from src.imu.windowing import make_sliding_windows


def build_grf_input_from_imu_csv(
    csv_path: Path,
    *,
    window_len: int = 256,
    stride: int | None = None,
) -> np.ndarray:
    """Build a GRF-model input tensor from a dual-IMU CSV.

    Expected model input shape (per src/vnext/models/vnext_fz.py and vnext_grf3d.py):
      x: (B, T, C) where:
        - B = number of windows
        - T = window_len
        - C = 12 channels in this exact order:
            ax_thigh, ay_thigh, az_thigh, gx_thigh, gy_thigh, gz_thigh,
            ax_shank, ay_shank, az_shank, gx_shank, gy_shank, gz_shank
    """

    csv_path = Path(csv_path)
    if stride is None:
        stride = int(window_len)

    rows = parse_imu_csv(csv_path)
    windowed = make_sliding_windows(
        rows,
        window_len=int(window_len),
        stride=int(stride),
        sensor_order=["thigh", "shank"],
    )

    expected = [
        "ax_thigh",
        "ay_thigh",
        "az_thigh",
        "gx_thigh",
        "gy_thigh",
        "gz_thigh",
        "ax_shank",
        "ay_shank",
        "az_shank",
        "gx_shank",
        "gy_shank",
        "gz_shank",
    ]
    if windowed.channel_names != expected:
        raise ValueError(
            "IMU window channel order mismatch; expected vNext canonical ordering. "
            f"expected={expected} got={windowed.channel_names}"
        )

    X = windowed.windows  # (B, C, T)
    X_btc = np.transpose(X, (0, 2, 1)).astype(np.float32, copy=False)  # (B, T, C)
    if not np.isfinite(X_btc).all():
        raise ValueError("Non-finite values in GRF input tensor")
    return X_btc
