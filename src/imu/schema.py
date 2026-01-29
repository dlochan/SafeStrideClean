from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import csv


@dataclass(frozen=True)
class IMURow:
    t_ms: int
    sensor_id: str

    ax: float
    ay: float
    az: float

    gx: float
    gy: float
    gz: float

    mag_x: Optional[float] = None
    mag_y: Optional[float] = None
    mag_z: Optional[float] = None


_REQUIRED = [
    "t_ms",
    "sensor_id",
    "ax", "ay", "az",
    "gx", "gy", "gz",
]

_OPTIONAL = ["mag_x", "mag_y", "mag_z"]


def _f(x: str) -> float:
    return float(x)


def _norm_col(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_").replace("-", "_")


_ALIASES = {
    "t_ms": ["t_ms", "time_ms", "timestamp_ms"],
    "sensor_id": ["sensor_id", "sensor", "device_id"],
    "ax": ["ax", "accel_x", "acc_x", "accx"],
    "ay": ["ay", "accel_y", "acc_y", "accy"],
    "az": ["az", "accel_z", "acc_z", "accz"],
    "gx": ["gx", "gyro_x", "gyr_x", "gyrx"],
    "gy": ["gy", "gyro_y", "gyr_y", "gyry"],
    "gz": ["gz", "gyro_z", "gyr_z", "gyrz"],
    "mag_x": ["mag_x", "mx"],
    "mag_y": ["mag_y", "my"],
    "mag_z": ["mag_z", "mz"],
}


def parse_imu_csv(path: str | Path) -> List[IMURow]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    rows: List[IMURow] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV missing header row")

        raw_cols = list(reader.fieldnames)
        cols_norm = [_norm_col(c) for c in raw_cols]
        if len(set(cols_norm)) != len(cols_norm):
            raise ValueError(
                "CSV header has duplicate columns after normalization. "
                f"raw={raw_cols} normalized={cols_norm}"
            )
        norm_to_raw = {n: r for n, r in zip(cols_norm, raw_cols)}

        def pick_col(canonical: str) -> Optional[str]:
            for cand in _ALIASES.get(canonical, [canonical]):
                if cand in norm_to_raw:
                    return norm_to_raw[cand]
            return None

        t_col = pick_col("t_ms")
        sensor_col = pick_col("sensor_id")
        if t_col is None:
            raise ValueError("CSV missing required column: t_ms")
        if sensor_col is None:
            raise ValueError("CSV missing required column: sensor_id")

        resolved = {c: pick_col(c) for c in _REQUIRED}
        missing = [c for c, raw in resolved.items() if raw is None]
        if missing:
            raise ValueError(
                "CSV missing required IMU columns. "
                f"Missing={missing}. "
                "Expected: t_ms,sensor_id,ax,ay,az,gx,gy,gz "
                "(aliases accepted: accel_x..accel_z, gyro_x..gyro_z)."
            )

        mag_cols = [pick_col(k) for k in _OPTIONAL]
        mag_present = all(c is not None for c in mag_cols)

        for i, r in enumerate(reader):
            try:
                t_raw = r.get(t_col)
                if t_raw in (None, ""):
                    raise ValueError("t_ms is blank")
                t_ms = int(t_raw)

                sensor_id = str(r.get(sensor_col, "")).strip()
                if not sensor_id:
                    raise ValueError("sensor_id is blank")

                def f_for(canonical: str) -> float:
                    raw = resolved[canonical]
                    assert raw is not None
                    v = r.get(raw)
                    if v in (None, ""):
                        raise ValueError(f"{canonical} is blank")
                    return _f(v)

                row = IMURow(
                    t_ms=t_ms,
                    sensor_id=sensor_id,
                    ax=f_for("ax"),
                    ay=f_for("ay"),
                    az=f_for("az"),
                    gx=f_for("gx"),
                    gy=f_for("gy"),
                    gz=f_for("gz"),
                    mag_x=_f(r[mag_cols[0]]) if mag_present and r.get(mag_cols[0]) not in (None, "") else None,
                    mag_y=_f(r[mag_cols[1]]) if mag_present and r.get(mag_cols[1]) not in (None, "") else None,
                    mag_z=_f(r[mag_cols[2]]) if mag_present and r.get(mag_cols[2]) not in (None, "") else None,
                )
            except Exception as e:
                raise ValueError(
                    f"Bad row {i} in {p.name}: {e}. "
                    "Expected numeric IMU columns ax,ay,az,gx,gy,gz and integer t_ms."
                ) from e
            rows.append(row)

    if not rows:
        raise ValueError("CSV contained zero data rows")

    # Simple sanity: allow interleaved streams, but require per-sensor time to be non-decreasing.
    last_t = {}  # sensor_id -> last seen t_ms
    for row_idx, r in enumerate(rows):
        prev = last_t.get(r.sensor_id)
        if prev is not None and r.t_ms < prev:
            raise ValueError(
                "t_ms must be non-decreasing within each sensor_id. "
                f"sensor_id={r.sensor_id} prev_t_ms={prev} curr_t_ms={r.t_ms} at_row={row_idx}"
            )
        last_t[r.sensor_id] = r.t_ms
    return rows
