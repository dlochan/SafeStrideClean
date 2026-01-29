from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .features import FeatureTensor, extract_features
from .schema import IMURow, parse_imu_csv
from .windowing import WindowedIMU, make_sliding_windows


@dataclass(frozen=True)
class IMUContractSummary:
    data: Dict[str, Any]


def compute_contract_summary(
    fixture_csv: str | Path,
    *,
    window_len: int,
    stride: int,
    include_magnitude: bool = False,
) -> Dict[str, Any]:
    rows: List[IMURow] = parse_imu_csv(fixture_csv)
    sensors = sorted({r.sensor_id for r in rows})

    windowed: WindowedIMU = make_sliding_windows(rows, window_len=window_len, stride=stride)
    feats: FeatureTensor = extract_features(windowed, include_magnitude=include_magnitude)

    # FeatureTensor.X is (N, C, L). Contract shape is (B, T, C).
    X_ncl = np.asarray(feats.X, dtype=np.float32)
    X_btc = np.transpose(X_ncl, (0, 2, 1))  # (B, T, C)

    has_nan = bool(np.isnan(X_btc).any() or np.isinf(X_btc).any())

    # Per-channel stats over (B,T)
    ch_min = X_btc.min(axis=(0, 1))
    ch_max = X_btc.max(axis=(0, 1))
    ch_mean = X_btc.mean(axis=(0, 1))
    ch_std = X_btc.std(axis=(0, 1))

    return {
        "units": {
            "t_ms": "ms",
            "accel": "m/s^2",
            "gyro": "rad/s",
            "mag": "unknown",
        },
        "window_len": int(window_len),
        "stride": int(stride),
        "num_windows": int(X_btc.shape[0]),
        "num_sensors": int(len(sensors)),
        "feature_tensor_shape": [int(d) for d in X_btc.shape],
        "has_nan": has_nan,
        "channel_names": list(feats.channel_names),
        "channel_min": [float(x) for x in ch_min],
        "channel_max": [float(x) for x in ch_max],
        "channel_mean": [float(x) for x in ch_mean],
        "channel_std": [float(x) for x in ch_std],
    }
