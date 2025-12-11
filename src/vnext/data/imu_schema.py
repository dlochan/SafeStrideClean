from __future__ import annotations

from typing import Dict, List

import pandas as pd

# Canonical dual-IMU schema for vNext scaffolding.
# One IMU approximately 2 inches above the knee (thigh),
# one approximately 2 inches below (shank).

TIME_COL = "time_s"

IMU_TAGS = ["thigh", "shank"]
AXES = ["x", "y", "z"]

ACC_PREFIX = "a"  # accelerometer
GYRO_PREFIX = "g"  # gyroscope


def _axis_cols(prefix: str, tag: str) -> List[str]:
    return [f"{prefix}{axis}_{tag}" for axis in AXES]


EXPECTED_IMU_COLUMNS: List[str] = [TIME_COL]
for _tag in IMU_TAGS:
    EXPECTED_IMU_COLUMNS.extend(_axis_cols("ax", _tag))
    EXPECTED_IMU_COLUMNS.extend(_axis_cols("gx", _tag))


def validate_canonical_imu_df(df: pd.DataFrame) -> None:
    """Validate that a DataFrame follows the canonical dual-IMU schema.

    Required columns:
    - time_s
    - ax_thigh, ay_thigh, az_thigh, gx_thigh, gy_thigh, gz_thigh
    - ax_shank, ay_shank, az_shank, gx_shank, gy_shank, gz_shank

    Raises
    ------
    ValueError
        If any required columns are missing.
    """

    missing = [c for c in EXPECTED_IMU_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Canonical IMU schema violation; missing columns: {missing}")


def get_feature_columns() -> List[str]:
    """Return ordered feature columns (IMU only, no time_s).

    This corresponds to EXPECTED_IMU_COLUMNS with the time column removed,
    and will be used as the canonical channel order for models.
    """

    return [c for c in EXPECTED_IMU_COLUMNS if c != TIME_COL]


def get_sensor_slices(feature_cols: List[str]) -> Dict[str, slice]:
    """Return contiguous channel slices for each sensor.

    Parameters
    ----------
    feature_cols:
        Ordered list of feature column names (no time_s), typically from
        get_feature_columns().

    Returns
    -------
    dict
        Mapping from sensor tag (e.g. "thigh", "shank") to a slice object
        that indexes the channel dimension of tensors shaped (T, C) or
        (B, T, C).
    """

    slices: Dict[str, slice] = {}
    for tag in IMU_TAGS:
        prefix_ax = f"ax_{tag}"
        prefix_gx = f"gx_{tag}"
        start = None
        end = None
        for idx, name in enumerate(feature_cols):
            if name.endswith(f"_{tag}"):
                if start is None:
                    start = idx
                end = idx + 1
        if start is not None and end is not None:
            slices[tag] = slice(start, end)
    return slices
