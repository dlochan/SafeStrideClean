from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

from .schema import IMURow


@dataclass(frozen=True)
class WindowedIMU:
    windows: np.ndarray  # (N, C, L)
    t0_ms: np.ndarray  # (N,)
    channel_names: List[str]


def _sensor_order_from_rows(rows: Sequence[IMURow]) -> List[str]:
    sensors = sorted({r.sensor_id for r in rows})
    if set(sensors) >= {"thigh", "shank"}:
        ordered = ["thigh", "shank"]
        for s in sensors:
            if s not in ordered:
                ordered.append(s)
        return ordered
    return sensors


def make_sliding_windows(
    rows: Iterable[IMURow],
    *,
    window_len: int,
    stride: int,
    sensor_order: Sequence[str] | None = None,
) -> WindowedIMU:
    rows_list = list(rows)
    if window_len <= 0:
        raise ValueError("window_len must be > 0")
    if stride <= 0:
        raise ValueError("stride must be > 0")
    if not rows_list:
        raise ValueError("rows must be non-empty")

    if sensor_order is None:
        sensor_order = _sensor_order_from_rows(rows_list)

    by_sensor: dict[str, list[IMURow]] = {}
    for r in rows_list:
        by_sensor.setdefault(r.sensor_id, []).append(r)

    missing = [s for s in sensor_order if s not in by_sensor]
    if missing:
        raise ValueError(f"sensor_order contains sensors not present in data: {missing}")

    # Sort each stream by time.
    sensor_ts: dict[str, list[int]] = {}
    sensor_X: dict[str, np.ndarray] = {}
    for s in sensor_order:
        rs = sorted(by_sensor[s], key=lambda r: r.t_ms)
        t = [r.t_ms for r in rs]
        X = np.asarray([[r.ax, r.ay, r.az, r.gx, r.gy, r.gz] for r in rs], dtype=np.float32)
        sensor_ts[s] = t
        sensor_X[s] = X

    # Align by common timestamps across all requested sensors.
    common = None
    for s in sensor_order:
        ts_set = set(sensor_ts[s])
        common = ts_set if common is None else common.intersection(ts_set)
    if not common:
        raise ValueError("No common timestamps across sensors; cannot align windows")

    common_ts = sorted(common)
    if len(common_ts) < window_len:
        raise ValueError(
            f"Not enough aligned samples for windowing: aligned_len={len(common_ts)} window_len={window_len}"
        )

    # Build aligned matrix (T, C_total).
    aligned_cols: list[np.ndarray] = []
    channel_names: list[str] = []
    for s in sensor_order:
        idx_by_t = {t: i for i, t in enumerate(sensor_ts[s])}
        idx = [idx_by_t[t] for t in common_ts]
        aligned_cols.append(sensor_X[s][idx])
        channel_names.extend([f"ax_{s}", f"ay_{s}", f"az_{s}", f"gx_{s}", f"gy_{s}", f"gz_{s}"])

    aligned = np.concatenate(aligned_cols, axis=1)  # (T, C_total)

    windows = []
    t0 = []
    for start in range(0, len(common_ts) - window_len + 1, stride):
        end = start + window_len
        w = aligned[start:end].T  # (C, L)
        windows.append(w)
        t0.append(common_ts[start])

    return WindowedIMU(
        windows=np.stack(windows, axis=0).astype(np.float32),
        t0_ms=np.asarray(t0, dtype=np.int64),
        channel_names=channel_names,
    )
