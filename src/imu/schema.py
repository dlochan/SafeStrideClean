from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import csv


@dataclass(frozen=True)
class IMURow:
    t_ms: int
    sensor_id: str

    accel_x: float
    accel_y: float
    accel_z: float

    gyro_x: float
    gyro_y: float
    gyro_z: float

    mag_x: Optional[float] = None
    mag_y: Optional[float] = None
    mag_z: Optional[float] = None


_REQUIRED = [
    "t_ms",
    "sensor_id",
    "accel_x", "accel_y", "accel_z",
    "gyro_x", "gyro_y", "gyro_z",
]

_OPTIONAL = ["mag_x", "mag_y", "mag_z"]


def _f(x: str) -> float:
    return float(x)


def parse_imu_csv(path: str | Path) -> List[IMURow]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    rows: List[IMURow] = []
    with p.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV missing header row")

        missing = [c for c in _REQUIRED if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        for i, r in enumerate(reader):
            try:
                mag_present = all((k in r and r[k] not in (None, "")) for k in _OPTIONAL)
                row = IMURow(
                    t_ms=int(r["t_ms"]),
                    sensor_id=str(r["sensor_id"]),
                    accel_x=_f(r["accel_x"]),
                    accel_y=_f(r["accel_y"]),
                    accel_z=_f(r["accel_z"]),
                    gyro_x=_f(r["gyro_x"]),
                    gyro_y=_f(r["gyro_y"]),
                    gyro_z=_f(r["gyro_z"]),
                    mag_x=_f(r["mag_x"]) if mag_present else None,
                    mag_y=_f(r["mag_y"]) if mag_present else None,
                    mag_z=_f(r["mag_z"]) if mag_present else None,
                )
            except Exception as e:
                raise ValueError(f"Bad row {i} in {p.name}: {e}") from e
            rows.append(row)

    if not rows:
        raise ValueError("CSV contained zero data rows")

# Simple sanity: require per-sensor time to be non-decreasing
    last_t = {}  # sensor_id -> last seen t_ms
    for r in rows:
        prev = last_t.get(r.sensor_id)
        if prev is not None and r.t_ms < prev:
            raise ValueError(f"t_ms must be non-decreasing within each sensor_id (sensor_id={r.sensor_id})")
        last_t[r.sensor_id] = r.t_ms
    return rows
