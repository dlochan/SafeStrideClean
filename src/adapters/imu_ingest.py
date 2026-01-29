from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from src.imu.features import extract_features
from src.imu.schema import parse_imu_csv
from src.imu.windowing import make_sliding_windows


def ingest_imu_csv(csv_path: Path, window_len: int, stride: int) -> Dict[str, Any]:
    rows = parse_imu_csv(csv_path)
    sensor_ids = sorted({r.sensor_id for r in rows})

    windowed = make_sliding_windows(rows, window_len=window_len, stride=stride)
    feats = extract_features(windowed, include_magnitude=False)

    # feats.X: (B, C, T) -> (B, T, C)
    X = np.transpose(np.asarray(feats.X, dtype=np.float32), (0, 2, 1))

    if X.ndim != 3:
        raise ValueError(f"X must have shape (B,T,C), got {X.shape}")
    b, t, c = X.shape
    if b <= 0 or t <= 0 or c <= 0:
        raise ValueError(f"X must be non-empty (B,T,C), got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN/inf")

    t0_ms = [int(x) for x in windowed.t0_ms.tolist()]

    sensor_map = {sid: {"index": i} for i, sid in enumerate(sensor_ids)}

    return {
        "meta": {
            "window_len": int(window_len),
            "stride": int(stride),
            "num_windows": int(b),
            "sensor_ids": list(sensor_ids),
            "num_sensors": int(len(sensor_ids)),
        },
        "X": X,
        "t0_ms": t0_ms,
        "sensor_map": sensor_map,
    }
