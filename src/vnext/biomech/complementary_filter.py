from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ComplementaryFilterConfig:
    alpha: float = 0.98
    accel_forward_idx: int = 0
    accel_up_idx: int = 1
    accel_forward_sign: float = 1.0
    accel_up_sign: float = 1.0
    gyro_pitch_idx: int = 1
    gyro_pitch_sign: float = 1.0


@dataclass(frozen=True)
class CanonicalIMUColumnMapping:
    accel_cols: Tuple[str, str, str]
    gyro_cols: Tuple[str, str, str]


CANONICAL_VNEXT_IMU_MAPPING = CanonicalIMUColumnMapping(
    accel_cols=("axx", "axy", "axz"),
    gyro_cols=("gxx", "gxy", "gxz"),
)


def pitch_from_accel(accel_t3: np.ndarray, cfg: ComplementaryFilterConfig) -> np.ndarray:
    a = np.asarray(accel_t3, dtype=float)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"accel must be shape (T,3), got {a.shape}")

    fwd = cfg.accel_forward_sign * a[:, int(cfg.accel_forward_idx)]
    up = cfg.accel_up_sign * a[:, int(cfg.accel_up_idx)]
    return np.arctan2(fwd, up)


def complementary_filter_pitch(
    accel_t3: np.ndarray,
    gyro_t3: np.ndarray,
    fs_hz: float,
    cfg: ComplementaryFilterConfig | None = None,
) -> np.ndarray:
    cfg = cfg or ComplementaryFilterConfig()

    a = np.asarray(accel_t3, dtype=float)
    g = np.asarray(gyro_t3, dtype=float)

    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"accel must be shape (T,3), got {a.shape}")
    if g.ndim != 2 or g.shape[1] != 3:
        raise ValueError(f"gyro must be shape (T,3), got {g.shape}")
    if a.shape[0] != g.shape[0]:
        raise ValueError(f"accel and gyro length mismatch: {a.shape[0]} vs {g.shape[0]}")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be > 0")

    alpha = float(cfg.alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")

    dt = 1.0 / float(fs_hz)

    pitch_acc = pitch_from_accel(a, cfg)

    w = cfg.gyro_pitch_sign * g[:, int(cfg.gyro_pitch_idx)]

    theta = np.zeros((a.shape[0],), dtype=float)
    theta[0] = float(pitch_acc[0])

    for i in range(1, theta.shape[0]):
        theta_gyro = theta[i - 1] + float(w[i]) * dt
        theta[i] = alpha * theta_gyro + (1.0 - alpha) * float(pitch_acc[i])

    return theta


def indices_from_feature_columns(feature_cols: Sequence[str], names: Sequence[str]) -> List[int]:
    idx = {n: i for i, n in enumerate(feature_cols)}
    out: List[int] = []
    for n in names:
        if n not in idx:
            raise ValueError(f"Required feature column missing: {n}")
        out.append(int(idx[n]))
    return out


def extract_sensor_accel_gyro_from_windows(
    X_btc: np.ndarray,
    feature_cols: Sequence[str],
    *,
    sensor_tag: str,
    mapping: CanonicalIMUColumnMapping,
) -> Tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X_btc)
    if X.ndim != 3:
        raise ValueError(f"X_btc must be rank-3 (B,T,C), got {X.shape}")

    a_names = [f"{c}_{sensor_tag}" for c in mapping.accel_cols]
    g_names = [f"{c}_{sensor_tag}" for c in mapping.gyro_cols]

    a_idx = indices_from_feature_columns(feature_cols, a_names)
    g_idx = indices_from_feature_columns(feature_cols, g_names)

    accel = X[:, :, a_idx].astype(np.float64, copy=False)
    gyro = X[:, :, g_idx].astype(np.float64, copy=False)

    return accel, gyro
