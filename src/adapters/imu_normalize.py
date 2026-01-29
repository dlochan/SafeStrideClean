from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.vnext.data.imu_schema import get_feature_columns


def _norm_col(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_").replace("-", "_")


_AXIS_ALIASES: Dict[Tuple[str, str], List[str]] = {
    ("a", "x"): ["ax", "a_x", "acc_x", "accel_x", "acceleration_x", "linearacc_x"],
    ("a", "y"): ["ay", "a_y", "acc_y", "accel_y", "acceleration_y", "linearacc_y"],
    ("a", "z"): ["az", "a_z", "acc_z", "accel_z", "acceleration_z", "linearacc_z"],
    ("g", "x"): ["gx", "g_x", "gyro_x", "gyr_x", "wx"],
    ("g", "y"): ["gy", "g_y", "gyro_y", "gyr_y", "wy"],
    ("g", "z"): ["gz", "g_z", "gyro_z", "gyr_z", "wz"],
}


def _extract_time_s(df: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
    time_col: Optional[str] = None
    for col in df.columns:
        n = _norm_col(col)
        if n in {"time_s", "t_s", "time_sec", "time_seconds"}:
            time_col = col
            break
        if n in {"time_ms", "timestamp_ms", "t_ms"}:
            time_col = col
            break
    if time_col is not None:
        n = _norm_col(time_col)
        raw = pd.to_numeric(df[time_col], errors="coerce")
        if n in {"time_ms", "timestamp_ms", "t_ms"}:
            ts = raw.astype(float) / 1000.0
        else:
            ts = raw.astype(float)
    else:
        n_rows = len(df)
        ts = pd.Series(np.arange(n_rows, dtype=float) * 0.01, index=df.index)
    df2 = df.copy()
    df2["__time_s__"] = ts
    df2 = df2.sort_values("__time_s__").reset_index(drop=True)
    ts_sorted = df2.pop("__time_s__")
    return ts_sorted.astype(float), df2


def _canonical_axis(prefix: str) -> Tuple[str, str]:
    if not prefix or len(prefix) < 3:
        raise ValueError(f"Unexpected canonical prefix: {prefix}")
    kind = prefix[0]
    if kind not in ("a", "g"):
        raise ValueError(f"Unexpected canonical kind in prefix: {prefix}")
    axis = prefix[-1]
    if axis not in ("x", "y", "z"):
        raise ValueError(f"Unexpected canonical axis in prefix: {prefix}")
    return kind, axis


def _select_source_column(
    norm_by_col: Dict[str, str], *, kind: str, axis: str, tag: str
) -> Optional[str]:
    aliases = _AXIS_ALIASES.get((kind, axis), [])
    for col, norm in norm_by_col.items():
        tokens = norm.split("_")
        if tag not in tokens:
            continue
        other = [t for t in tokens if t != tag]
        joined = "_".join(other)
        for alias in aliases:
            if alias == joined or alias in other:
                return col
    return None


def normalize_imu_csv(
    input_csv_path: str, output_csv_path: Optional[str] = None
) -> pd.DataFrame:
    src_path = Path(input_csv_path)
    if not src_path.exists():
        raise FileNotFoundError(str(src_path))

    raw_df = pd.read_csv(src_path)
    time_s, df = _extract_time_s(raw_df)

    feature_cols = get_feature_columns()
    norm_by_col = {col: _norm_col(col) for col in df.columns}

    data: Dict[str, pd.Series] = {}
    missing: List[str] = []

    for name in feature_cols:
        if "_" not in name:
            missing.append(name)
            continue
        prefix, tag = name.split("_", 1)
        kind, axis = _canonical_axis(prefix)
        src_col = _select_source_column(norm_by_col, kind=kind, axis=axis, tag=tag)
        if src_col is None:
            missing.append(name)
            continue
        series = pd.to_numeric(df[src_col], errors="coerce")
        data[name] = series.astype(np.float32)

    if missing:
        raise ValueError(
            "Missing required canonical IMU channels: " + ", ".join(sorted(missing))
        )

    out = pd.DataFrame({"time_s": time_s})
    for name in feature_cols:
        out[name] = data[name].astype(np.float32)

    if output_csv_path is not None:
        dst = Path(output_csv_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(dst, index=False)

    return out
