from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class NormalizationDebug:
    raw_columns: List[str]
    canon_columns: List[str]
    used_aliases: List[Tuple[str, str]]
    dropped_columns: List[str]
    missing_canon_columns: List[str]


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


def _candidate_sources_for_canon(
    norm_by_col: Dict[str, str], *, kind: str, axis: str, tag: str
) -> List[str]:
    aliases = _AXIS_ALIASES.get((kind, axis), [])
    cands: List[str] = []
    for col, norm in norm_by_col.items():
        tokens = norm.split("_")
        if tag not in tokens:
            continue
        other = [t for t in tokens if t != tag]
        joined = "_".join(other)
        for alias in aliases:
            if alias == joined or alias in other:
                cands.append(col)
                break
    return cands


def _normalize_to_canon_internal(
    input_csv_path: str,
) -> Tuple[pd.DataFrame, NormalizationDebug]:
    src_path = Path(input_csv_path)
    if not src_path.exists():
        raise FileNotFoundError(str(src_path))

    raw_df = pd.read_csv(src_path)
    raw_columns = list(raw_df.columns)

    feature_cols = get_feature_columns()
    norm_by_col = {col: _norm_col(col) for col in raw_columns}

    data: Dict[str, pd.Series] = {}
    used_aliases: List[Tuple[str, str]] = []
    used_raw_cols: set[str] = set()
    missing_canon: List[str] = []

    for name in feature_cols:
        if "_" not in name:
            missing_canon.append(name)
            continue
        prefix, tag = name.split("_", 1)
        kind, axis = _canonical_axis(prefix)
        candidates = _candidate_sources_for_canon(
            norm_by_col, kind=kind, axis=axis, tag=tag
        )
        if not candidates:
            missing_canon.append(name)
            continue

        combined = np.full(len(raw_df), np.nan, dtype=np.float32)
        for raw_col in candidates:
            series = pd.to_numeric(raw_df[raw_col], errors="coerce").astype(
                np.float32
            )
            vals = series.to_numpy(copy=False)
            vals[~np.isfinite(vals)] = np.nan
            mask = np.isnan(combined) & ~np.isnan(vals)
            if mask.any():
                combined[mask] = vals[mask]
            used_raw_cols.add(raw_col)
            used_aliases.append((raw_col, name))

        data[name] = pd.Series(combined, index=raw_df.index, dtype=np.float32)

    debug = NormalizationDebug(
        raw_columns=raw_columns,
        canon_columns=feature_cols,
        used_aliases=used_aliases,
        dropped_columns=[c for c in raw_columns if c not in used_raw_cols],
        missing_canon_columns=sorted(missing_canon),
    )

    if debug.missing_canon_columns:
        raise ValueError(
            "Missing required canonical IMU channels: "
            + ", ".join(debug.missing_canon_columns)
        )

    values = np.column_stack(
        [data[name].to_numpy(dtype=np.float32, copy=False) for name in feature_cols]
    )
    if not np.isfinite(values).all():
        raise ValueError("non-finite after normalization")

    out = pd.DataFrame({name: data[name].astype(np.float32) for name in feature_cols})
    return out, debug


def normalize_imu_csv_to_canon_df(input_csv_path: str) -> pd.DataFrame:
    df, _debug = _normalize_to_canon_internal(input_csv_path)
    return df


def normalize_imu_csv_to_canon_df_with_debug(
    input_csv_path: str,
) -> Tuple[pd.DataFrame, NormalizationDebug]:
    return _normalize_to_canon_internal(input_csv_path)


def normalize_imu_csv(
    input_csv_path: str, output_csv_path: Optional[str] = None
) -> pd.DataFrame:
    df, _debug = _normalize_to_canon_internal(input_csv_path)
    if output_csv_path is not None:
        dst = Path(output_csv_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(dst, index=False)
    return df
