from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class KneeStanceDetection2DConfig:
    fz_threshold_n: float = 50.0
    min_stance_duration_s: float = 0.20
    g: float = 9.81


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 1:
        return x[None, :]
    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")
    return x


def _contiguous_true_segments(mask_1d: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(mask_1d, dtype=bool)
    if mask.size == 0:
        return []

    x = mask.astype(np.int8)
    dx = np.diff(x)
    starts = list(np.where(dx == 1)[0] + 1)
    ends = list(np.where(dx == -1)[0] + 1)

    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [mask.size]

    if len(starts) != len(ends):
        return []

    return [(int(s), int(e)) for s, e in zip(starts, ends) if int(e) > int(s)]


def detect_stance_2d(
    fz_n: np.ndarray,
    *,
    fs_hz: float,
    cfg: KneeStanceDetection2DConfig | None = None,
) -> Tuple[np.ndarray, List[Optional[Tuple[int, int]]]]:
    cfg = cfg or KneeStanceDetection2DConfig()

    fz = _as_2d(np.asarray(fz_n, dtype=float))
    if fs_hz <= 0:
        raise ValueError("fs_hz must be > 0")

    thr = float(cfg.fz_threshold_n)
    raw = fz > thr

    min_len = int(np.ceil(float(cfg.min_stance_duration_s) * float(fs_hz)))
    min_len = max(min_len, 1)

    B, T = raw.shape
    stance = np.zeros((B, T), dtype=bool)
    segments: List[Optional[Tuple[int, int]]] = []

    for b in range(B):
        segs = _contiguous_true_segments(raw[b])
        segs = [(s, e) for (s, e) in segs if (e - s) >= min_len]
        if not segs:
            segments.append(None)
            continue

        segs = sorted(segs, key=lambda se: (-(se[1] - se[0]), se[0]))
        s0, e0 = segs[0]
        stance[b, s0:e0] = True
        segments.append((s0, e0))

    return stance, segments


def _iqr(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, 75.0) - np.percentile(x, 25.0))


def _cv(x: np.ndarray) -> float:
    if x.size == 0:
        return float("nan")
    m = float(np.mean(x))
    s = float(np.std(x))
    if not np.isfinite(m) or not np.isfinite(s):
        return float("nan")
    denom = abs(m) + 1e-12
    return float(s / denom)


def compute_knee_metrics_2d(
    fz_n: np.ndarray,
    knee_moment_nm_per_kg: np.ndarray,
    *,
    fs_hz: float,
    cfg: KneeStanceDetection2DConfig | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cfg = cfg or KneeStanceDetection2DConfig()

    fz = _as_2d(np.asarray(fz_n, dtype=float))
    m = _as_2d(np.asarray(knee_moment_nm_per_kg, dtype=float))

    if fz.shape != m.shape:
        raise ValueError(f"fz_n and knee_moment_nm_per_kg must match shape, got {fz.shape} vs {m.shape}")
    if fs_hz <= 0:
        raise ValueError("fs_hz must be > 0")

    dt = 1.0 / float(fs_hz)

    stance_mask, segments = detect_stance_2d(fz, fs_hz=float(fs_hz), cfg=cfg)

    per_window: List[Dict[str, Any]] = []

    for b in range(int(fz.shape[0])):
        seg = segments[b]
        if seg is None:
            raise ValueError(f"No stance detected in window {b} (threshold={cfg.fz_threshold_n}, min_dur={cfg.min_stance_duration_s})")
        s, e = seg
        if e <= s:
            raise ValueError(f"Invalid stance segment in window {b}: {(s, e)}")

        fz_st = fz[b, s:e]
        m_st = m[b, s:e]

        if not np.isfinite(fz_st).all() or not np.isfinite(m_st).all():
            raise ValueError(f"Non-finite values inside stance segment for window {b}")

        g = float(cfg.g)
        if g <= 0:
            raise ValueError("cfg.g must be > 0")
        mass_est_kg = float(np.median(fz_st) / g)
        if not np.isfinite(mass_est_kg) or mass_est_kg <= 0:
            raise ValueError(f"Invalid mass estimate in window {b}: {mass_est_kg}")

        fz_n_per_kg = fz_st / mass_est_kg

        peak_fz_n_per_kg = float(np.max(fz_n_per_kg))
        impulse_fz_ns_per_kg = float(np.sum(fz_n_per_kg) * dt)

        dfdt = np.gradient(fz_n_per_kg, dt)
        peak_fz_i = int(np.argmax(fz_n_per_kg))
        if peak_fz_i <= 0:
            loading_rate = float(np.max(dfdt))
        else:
            loading_rate = float(np.max(dfdt[: peak_fz_i + 1]))

        abs_m = np.abs(m_st)
        peak_moment_nm_per_kg = float(np.max(abs_m))
        peak_m_i = int(np.argmax(abs_m))
        time_to_peak_moment_s = float(peak_m_i * dt)
        moment_impulse_nms_per_kg = float(np.sum(m_st) * dt)

        per_window.append(
            {
                "window_index": int(b),
                "stance_start_idx": int(s),
                "stance_end_idx": int(e),
                "stance_duration_s": float((e - s) * dt),
                "mass_est_kg": float(mass_est_kg),
                "peak_fz_n_per_kg": float(peak_fz_n_per_kg),
                "impulse_fz_ns_per_kg": float(impulse_fz_ns_per_kg),
                "loading_rate_n_per_kg_s": float(loading_rate),
                "peak_knee_moment_nm_per_kg": float(peak_moment_nm_per_kg),
                "moment_impulse_nms_per_kg": float(moment_impulse_nms_per_kg),
                "time_to_peak_moment_s": float(time_to_peak_moment_s),
            }
        )

    def arr(key: str) -> np.ndarray:
        return np.asarray([float(r[key]) for r in per_window], dtype=np.float64)

    peak_fz = arr("peak_fz_n_per_kg")
    peak_m = arr("peak_knee_moment_nm_per_kg")

    summary: Dict[str, Any] = {
        "fs_hz": float(fs_hz),
        "stance_cfg": asdict(cfg),
        "n_windows": int(len(per_window)),
        "peak_fz_n_per_kg": {
            "median": float(np.median(peak_fz)),
            "iqr": _iqr(peak_fz),
            "cv": _cv(peak_fz),
        },
        "peak_knee_moment_nm_per_kg": {
            "median": float(np.median(peak_m)),
            "iqr": _iqr(peak_m),
            "cv": _cv(peak_m),
        },
    }

    for k in [
        "impulse_fz_ns_per_kg",
        "loading_rate_n_per_kg_s",
        "moment_impulse_nms_per_kg",
        "time_to_peak_moment_s",
    ]:
        x = arr(k)
        summary[k] = {"median": float(np.median(x)), "iqr": _iqr(x), "cv": _cv(x)}

    return per_window, summary
