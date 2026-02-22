from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class KneeMoment2DConfig:
    g: float = 9.81

    shank_mass_fraction: float = 0.0465
    shank_com_fraction_from_knee: float = 0.433
    inertia_coeff_k: float = 0.10

    theta_lowpass_cutoff_hz: Optional[float] = 6.0
    theta_lowpass_order: int = 4

    savgol_window_length: int = 21
    savgol_polyorder: int = 3

    moment_lowpass_cutoff_hz: Optional[float] = None
    moment_lowpass_order: int = 4

    x_grf_from_ankle_m: float = 0.0

    enforce_theta_radians: bool = True
    max_abs_theta_rad: float = 6.5

    normalize_by_body_mass: bool = True
    peak_moment_bounds_nm_per_kg: Tuple[float, float] = (-3.0, 3.0)

    sign_flip: float = 1.0


@dataclass(frozen=True)
class KneeMoment2DResult:
    moment: np.ndarray
    moment_filtered: np.ndarray
    theta: np.ndarray
    theta_filtered: np.ndarray
    omega: np.ndarray
    alpha: np.ndarray
    terms: Dict[str, np.ndarray]
    metadata: Dict[str, Any]


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")
    return x


def _ema_lowpass(x: np.ndarray, fs: float, cutoff_hz: Optional[float]) -> np.ndarray:
    if cutoff_hz is None:
        return x
    if fs <= 0:
        return x
    if cutoff_hz <= 0:
        return x

    dt = 1.0 / float(fs)
    rc = 1.0 / (2.0 * np.pi * float(cutoff_hz))
    a = float(dt / (rc + dt))

    y = np.array(x, dtype=float, copy=True)
    if y.shape[-1] <= 1:
        return y

    y[..., 0] = x[..., 0]
    for i in range(1, y.shape[-1]):
        y[..., i] = a * x[..., i] + (1.0 - a) * y[..., i - 1]
    return y


def _finite_diff_derivatives(theta: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    dt = 1.0 / float(fs)
    omega = np.gradient(theta, dt, axis=-1)
    alpha = np.gradient(omega, dt, axis=-1)
    return omega, alpha


def _rotate(vec_xy: np.ndarray, theta: np.ndarray) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    x = vec_xy[..., 0]
    y = vec_xy[..., 1]
    xr = x * c - y * s
    yr = x * s + y * c
    return np.stack([xr, yr], axis=-1)


def _validate_theta(theta: np.ndarray, cfg: KneeMoment2DConfig) -> None:
    if not cfg.enforce_theta_radians:
        return
    mx = float(np.nanmax(np.abs(theta)))
    if not np.isfinite(mx):
        raise ValueError("theta contains no finite values")
    if mx > float(cfg.max_abs_theta_rad):
        raise ValueError(f"theta appears non-radian (max_abs={mx:.6g})")


def _validate_peak(moment_nm_per_kg: np.ndarray, cfg: KneeMoment2DConfig) -> None:
    lo, hi = cfg.peak_moment_bounds_nm_per_kg
    peak = float(np.nanmax(np.abs(moment_nm_per_kg)))
    if not np.isfinite(peak):
        raise ValueError("moment contains no finite values")
    if peak > max(abs(lo), abs(hi)):
        raise ValueError(f"peak moment {peak:.6g} Nm/kg outside bounds [{lo:.6g}, {hi:.6g}]")


def estimate_knee_moment_2d(
    theta_shank_rad: np.ndarray,
    fz_n: np.ndarray,
    *,
    fs_hz: float,
    body_mass_kg: float,
    l_shank_m: float,
    cfg: KneeMoment2DConfig | None = None,
) -> KneeMoment2DResult:
    cfg = cfg or KneeMoment2DConfig()

    theta = _as_2d(np.asarray(theta_shank_rad, dtype=float))
    fz = _as_2d(np.asarray(fz_n, dtype=float))

    if theta.shape != fz.shape:
        raise ValueError(f"theta and fz must match shape, got {theta.shape} vs {fz.shape}")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be > 0")
    if body_mass_kg <= 0:
        raise ValueError("body_mass_kg must be > 0")
    if l_shank_m <= 0:
        raise ValueError("l_shank_m must be > 0")

    _validate_theta(theta, cfg)

    theta_f = _ema_lowpass(theta, fs=float(fs_hz), cutoff_hz=cfg.theta_lowpass_cutoff_hz)
    omega, alpha = _finite_diff_derivatives(theta_f, fs=float(fs_hz))

    m_shank = float(cfg.shank_mass_fraction) * float(body_mass_kg)
    r_com = float(cfg.shank_com_fraction_from_knee) * float(l_shank_m)
    i_shank = float(cfg.inertia_coeff_k) * m_shank * (float(l_shank_m) ** 2)

    r_com_local = np.zeros(theta.shape + (2,), dtype=float)
    r_com_local[..., 1] = -r_com

    r_grf_local = np.zeros(theta.shape + (2,), dtype=float)
    r_grf_local[..., 0] = float(cfg.x_grf_from_ankle_m)
    r_grf_local[..., 1] = -float(l_shank_m)

    r_com_xy = _rotate(r_com_local, theta_f)
    r_grf_xy = _rotate(r_grf_local, theta_f)

    m_grf = r_grf_xy[..., 0] * fz
    m_g = r_com_xy[..., 0] * (-m_shank * float(cfg.g))
    m_inertia = i_shank * alpha

    m_knee = (m_inertia + m_g + m_grf) * float(cfg.sign_flip)
    m_knee_f = _ema_lowpass(m_knee, fs=float(fs_hz), cutoff_hz=cfg.moment_lowpass_cutoff_hz)

    m_knee_f_per_kg = m_knee_f / float(body_mass_kg)
    _validate_peak(m_knee_f_per_kg, cfg)

    if cfg.normalize_by_body_mass:
        moment = m_knee / float(body_mass_kg)
        moment_f = m_knee_f_per_kg
        terms = {
            "M_inertia_nm_per_kg": (m_inertia * float(cfg.sign_flip)) / float(body_mass_kg),
            "M_gravity_nm_per_kg": (m_g * float(cfg.sign_flip)) / float(body_mass_kg),
            "M_grf_nm_per_kg": (m_grf * float(cfg.sign_flip)) / float(body_mass_kg),
        }
        units = "Nm/kg"
    else:
        moment = m_knee
        moment_f = m_knee_f
        terms = {
            "M_inertia_nm": m_inertia * float(cfg.sign_flip),
            "M_gravity_nm": m_g * float(cfg.sign_flip),
            "M_grf_nm": m_grf * float(cfg.sign_flip),
        }
        units = "Nm"

    metadata: Dict[str, Any] = {
        "fs_hz": float(fs_hz),
        "body_mass_kg": float(body_mass_kg),
        "l_shank_m": float(l_shank_m),
        "m_shank_kg": float(m_shank),
        "r_com_m": float(r_com),
        "i_shank_kg_m2": float(i_shank),
        "moment_units": units,
        "cfg": asdict(cfg),
    }

    return KneeMoment2DResult(
        moment=moment,
        moment_filtered=moment_f,
        theta=theta,
        theta_filtered=theta_f,
        omega=omega,
        alpha=alpha,
        terms=terms,
        metadata=metadata,
    )
