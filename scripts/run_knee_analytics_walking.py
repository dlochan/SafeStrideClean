#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_sys_path() -> None:
    root = _repo_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def _git_short_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=str(_repo_root())
        )
        return out.decode("utf-8").strip()
    except Exception:
        return "unknown"


@dataclass(frozen=True)
class LongIMURow:
    t_ms: int
    sensor_id: str
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


def _read_long_imu_csv(path: Path) -> List[LongIMURow]:
    rows: List[LongIMURow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for rr in r:
            if not rr:
                continue
            t_ms = int(rr["t_ms"])
            sensor_id = str(rr["sensor_id"]).strip()
            rows.append(
                LongIMURow(
                    t_ms=t_ms,
                    sensor_id=sensor_id,
                    ax=float(rr["ax"]),
                    ay=float(rr["ay"]),
                    az=float(rr["az"]),
                    gx=float(rr["gx"]),
                    gy=float(rr["gy"]),
                    gz=float(rr["gz"]),
                )
            )
    if not rows:
        raise ValueError("empty imu csv")
    return rows


def _infer_sample_hz(rows: List[LongIMURow], default_hz: float) -> float:
    ts = sorted({int(r.t_ms) for r in rows})
    if len(ts) < 2:
        return float(default_hz)
    diffs = [ts[i + 1] - ts[i] for i in range(len(ts) - 1) if ts[i + 1] - ts[i] > 0]
    if not diffs:
        return float(default_hz)
    dt_ms = float(np.median(np.asarray(diffs, dtype=np.float64)))
    if dt_ms <= 0.0 or not np.isfinite(dt_ms):
        return float(default_hz)
    hz = 1000.0 / dt_ms
    if hz <= 0.0 or not np.isfinite(hz):
        return float(default_hz)
    return float(hz)


def _write_wide_csv_for_normalizer(rows: List[LongIMURow], out_path: Path) -> None:
    by_t: Dict[int, Dict[str, LongIMURow]] = {}
    for r in rows:
        by_t.setdefault(int(r.t_ms), {})[str(r.sensor_id)] = r

    ts_sorted = sorted(by_t.keys())
    sensors = ["thigh", "shank"]
    axes = ["ax", "ay", "az", "gx", "gy", "gz"]
    fieldnames = ["time_ms"] + [f"{a}_{s}" for s in sensors for a in axes]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t_ms in ts_sorted:
            d: Dict[str, Any] = {"time_ms": int(t_ms)}
            for s in sensors:
                rr = by_t.get(int(t_ms), {}).get(s)
                if rr is None:
                    for a in axes:
                        d[f"{a}_{s}"] = ""
                else:
                    d[f"ax_{s}"] = f"{float(rr.ax):.6f}"
                    d[f"ay_{s}"] = f"{float(rr.ay):.6f}"
                    d[f"az_{s}"] = f"{float(rr.az):.6f}"
                    d[f"gx_{s}"] = f"{float(rr.gx):.6f}"
                    d[f"gy_{s}"] = f"{float(rr.gy):.6f}"
                    d[f"gz_{s}"] = f"{float(rr.gz):.6f}"
            w.writerow(d)


def _write_curve_csv(path: Path, curve: np.ndarray) -> None:
    x = np.asarray(curve, dtype=np.float32).reshape(-1)
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("idx,moment_nm_per_kg\n")
        for i, v in enumerate(x.tolist()):
            f.write(f"{int(i)},{float(v):.6f}\n")


def _write_fz_csv(path: Path, fz_n: np.ndarray, *, mass_kg: float) -> None:
    x = np.asarray(fz_n, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise ValueError("empty fz")
    m = float(mass_kg)
    if m <= 0.0 or not np.isfinite(m):
        raise ValueError("invalid mass_kg")
    fz_n_per_kg = (x.astype(np.float64) / m).astype(np.float32, copy=False)

    with path.open("w", encoding="utf-8", newline="") as f:
        f.write("idx,fz_n,fz_n_per_kg\n")
        for i, (v1, v2) in enumerate(zip(x.tolist(), fz_n_per_kg.tolist())):
            f.write(f"{int(i)},{float(v1):.6f},{float(v2):.6f}\n")


def _write_plot(path: Path, curve: np.ndarray) -> None:
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("matplotlib is required to write knee_moment_plot.png") from e

    x = np.asarray(curve, dtype=np.float32).reshape(-1)
    fig = plt.figure(figsize=(7, 3.5), dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.arange(x.size), x, linewidth=2.0)
    ax.set_xlabel("idx")
    ax.set_ylabel("moment_nm_per_kg")
    ax.set_title("knee moment proxy walking")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(path))
    plt.close(fig)


def _stance_window_from_fz(
    fz_n: np.ndarray,
    *,
    body_mass_kg: float,
    sample_hz: float,
    g: float = 9.81,
    threshold_n_per_kg: float = 1.0,
    smooth_window_samples: int = 5,
) -> Dict[str, Any]:
    x = np.asarray(fz_n, dtype=np.float32).reshape(-1)
    if x.size == 0:
        raise ValueError("empty fz")
    if not np.isfinite(x).all():
        raise ValueError("non-finite fz")

    mass = float(body_mass_kg)
    if mass <= 0.0 or not np.isfinite(mass):
        raise ValueError("invalid body_mass_kg")
    hz = float(sample_hz)
    if hz <= 0.0 or not np.isfinite(hz):
        raise ValueError("invalid sample_hz")

    fz_n_per_kg = x.astype(np.float64) / float(mass)

    w = int(smooth_window_samples)
    if w < 1:
        raise ValueError("invalid smooth_window_samples")
    if w == 1:
        fz_sm = fz_n_per_kg
    else:
        pad_l = w // 2
        pad_r = int(w - 1 - pad_l)
        xp = np.pad(fz_n_per_kg, (pad_l, pad_r), mode="edge")
        kernel = (np.ones(w, dtype=np.float64) / float(w)).astype(np.float64, copy=False)
        fz_sm = np.convolve(xp, kernel, mode="valid")
        if fz_sm.shape != fz_n_per_kg.shape:
            raise RuntimeError("unexpected smoothing shape")

    thr = float(threshold_n_per_kg)
    contact = fz_sm > thr
    contact_fraction_mask = float(np.sum(contact)) / float(contact.size)

    segments: List[Tuple[int, int]] = []
    i = 0
    n = int(contact.size)
    while i < n:
        if not bool(contact[i]):
            i += 1
            continue
        j = i + 1
        while j < n and bool(contact[j]):
            j += 1
        segments.append((int(i), int(j - 1)))
        i = j

    method = "longest_contiguous_contact_segment"
    raw_start_idx = 0
    raw_end_idx = int(x.size - 1)

    if not segments:
        method = "fallback_full_window_contact_always_true"
        start_idx = 0
        end_idx = int(x.size - 1)
    else:
        seg_lens = [int(e - s + 1) for (s, e) in segments]
        best_i = int(np.argmax(np.asarray(seg_lens, dtype=np.int64)))
        start_idx, end_idx = segments[best_i]

        raw_start_idx = int(start_idx)
        raw_end_idx = int(end_idx)

        if int(end_idx - start_idx + 1) == int(x.size):
            method = "fallback_peak_centered_window_due_to_full_contact_mask"
            max_stance_duration_s = 0.08
            cap_len = int(round(float(max_stance_duration_s) * float(hz)))
            cap_len = max(3, min(int(x.size), int(cap_len)))
            peak_idx0 = int(np.argmax(fz_sm))
            start_idx = int(max(0, peak_idx0 - (cap_len // 2)))
            end_idx = int(start_idx + cap_len - 1)
            if end_idx > int(x.size - 1):
                end_idx = int(x.size - 1)
                start_idx = int(end_idx - cap_len + 1)

    seg = fz_sm[int(start_idx) : int(end_idx) + 1]
    peak_idx = int(start_idx + int(np.argmax(seg)))
    peak_fz_n_per_kg = float(fz_sm[peak_idx])

    contact_fraction = float(int(end_idx - start_idx + 1)) / float(x.size)

    midstance_idx = int((start_idx + end_idx) // 2)
    duration_s = float(end_idx - start_idx + 1) / float(hz)
    bw_n = float(mass) * float(g)

    return {
        "method": str(method),
        "threshold_n_per_kg": float(thr),
        "contact_fraction": float(contact_fraction),
        "contact_fraction_mask": float(contact_fraction_mask),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
        "raw_start_idx": int(raw_start_idx),
        "raw_end_idx": int(raw_end_idx),
        "peak_fz_idx": int(peak_idx),
        "peak_fz_n_per_kg": float(peak_fz_n_per_kg),
        "midstance_idx": int(midstance_idx),
        "duration_s": float(duration_s),
        "bw_n": float(bw_n),
    }


def _artifact_readme_text(*, knee_metrics: Dict[str, Any]) -> str:
    u = knee_metrics.get("units", {})
    fz_conv = ((knee_metrics.get("inputs") or {}).get("fz_units")) or {}
    stance = knee_metrics.get("stance", {})
    dt_s = ((knee_metrics.get("inputs") or {}).get("dt_s"))
    sample_hz = ((knee_metrics.get("inputs") or {}).get("sample_hz"))
    smooth_window_samples = ((knee_metrics.get("inputs") or {}).get("smooth_window_samples"))

    return (
        "# Knee analytics (walking-only) artifacts\n"
        "\n"
        "This README is generated per-run under artifacts/knee_analytics_walk_*/README.md.\n"
        "\n"
        "## Inputs\n"
        "- Canonical IMU: derived from the walking fixture and normalized via the repo IMU normalizer.\n"
        "- Predicted vertical GRF (Fz): inferred by the IMU->GRF path (vNext Fz model or deterministic fallback).\n"
        "\n"
        "### Units\n"
        f"- Fz curve: {u.get('fz_curve', 'unknown')}\n"
        f"- Fz curve (per kg): {u.get('fz_curve_per_kg', 'unknown')}\n"
        f"- Knee moment curve: {u.get('knee_moment_curve', 'unknown')}\n"
        "\n"
        "### Fz conversion provenance\n"
        f"- Converted to Newtons: {fz_conv.get('units', 'unknown')}\n"
        f"- target_norm.json: {fz_conv.get('target_norm_path', 'unknown')}\n"
        f"- target_norm.json sha256: {fz_conv.get('target_norm_json_sha256', 'unknown')}\n"
        "\n"
        "## Outputs\n"
        "- fz_curve.csv: predicted vertical GRF curve for the first window (both N and N/kg columns).\n"
        "- knee_moment_curve.csv: proxy knee sagittal-plane moment curve for the first window.\n"
        "- knee_moment_plot.png: plot of the moment curve.\n"
        "- knee_metrics.json: summary stats and provenance for this run.\n"
        "- provenance.txt: minimal run provenance (git sha, parameters).\n"
        "\n"
        "### What the knee moment curve means\n"
        "This is a proxy: moment(t) = Fz(t) * lever_arm_m, normalized by body mass to yield Nm/kg.\n"
        "It is meant to provide a stable, interpretable signal for a walking-only knee health slice.\n"
        "\n"
        "### Sanity checks (warn-only)\n"
        "Sanity check bands are MVP guardrails for debugging and CI monitoring; they are not validated clinical thresholds.\n"
        "\n"
        "### Stance window assumptions\n"
        "We define a stance window within the 256-sample window by thresholding smoothed Fz (N/kg) and\n"
        "selecting the longest contiguous above-threshold segment. Mid-stance is identified as the midpoint\n"
        "of the stance interval.\n"
        f"- stance method: {stance.get('method', 'unknown')}\n"
        f"- threshold_n_per_kg: {stance.get('threshold_n_per_kg', 'unknown')}\n"
        f"- stance start_idx: {stance.get('start_idx', 'unknown')}\n"
        f"- stance end_idx: {stance.get('end_idx', 'unknown')}\n"
        f"- peak_fz_idx: {stance.get('peak_fz_idx', 'unknown')}\n"
        f"- midstance_idx: {stance.get('midstance_idx', 'unknown')}\n"
        "\n"
        "### Smoothing and sampling\n"
        f"- sample_hz: {sample_hz}\n"
        f"- dt_s: {dt_s}\n"
        f"- smooth_window_samples: {smooth_window_samples}\n"
        "The knee moment curve is smoothed to avoid noise amplification.\n"
        "\n"
        "## Known limitations\n"
        "- 2D simplification: sagittal-plane proxy only.\n"
        "- Fz-only: no shear forces and no joint kinematics, so this is not a full inverse dynamics solution.\n"
        "- No subject-specific segment lengths yet; lever arm is a fixed proxy.\n"
        "- Predicted Fz depends on the IMU->GRF model and its normalization provenance.\n"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mass-kg", type=float, default=75.0)
    p.add_argument("--sample-hz", type=float, default=0.0)
    p.add_argument("--lever-arm-m", type=float, default=0.04)
    args = p.parse_args()

    _ensure_sys_path()

    fixture_rel = Path("tests/fixtures/imu_sample.csv")
    fixture_path = _repo_root() / fixture_rel
    if not fixture_path.exists():
        raise FileNotFoundError(str(fixture_path))

    gitsha = _git_short_sha()
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    out_dir = _repo_root() / "artifacts" / f"knee_analytics_walk_{ts}_{gitsha}"
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows = _read_long_imu_csv(fixture_path)

    from src.vnext.analytics.knee_analytics import compute_knee_moment_from_fz, summarize_curve

    sample_hz = float(args.sample_hz)
    if sample_hz <= 0.0:
        from src.vnext.analytics.knee_analytics import DEFAULT_SAMPLE_HZ

        sample_hz = _infer_sample_hz(long_rows, float(DEFAULT_SAMPLE_HZ))

    with tempfile.TemporaryDirectory() as tmpdir:
        wide_csv_path = Path(tmpdir) / "imu_wide_for_normalize.csv"
        _write_wide_csv_for_normalizer(long_rows, wide_csv_path)

        from src.adapters.imu_normalize import normalize_imu_csv_to_canon_df_with_debug

        _canon_df, debug = normalize_imu_csv_to_canon_df_with_debug(str(wide_csv_path))

    from scripts import check_imu_to_grf_nonregression as imu_to_grf_contract

    y = imu_to_grf_contract._run_inference()
    if y.shape != (64, 256, 1):
        raise ValueError(f"unexpected imu_to_grf output shape {y.shape}")

    from src.vnext.biomech.fz_units import to_newtons

    fz_n, fz_prov = to_newtons(y, body_mass_kg=float(args.mass_kg))
    fz_first = np.asarray(fz_n[0, :, 0], dtype=np.float32)
    fz_first_n_per_kg = (fz_first.astype(np.float64) / float(args.mass_kg)).astype(np.float32, copy=False)

    analytics = compute_knee_moment_from_fz(
        fz_first,
        mass_kg=float(args.mass_kg),
        sample_hz=float(sample_hz),
        stride_len_m=float(args.lever_arm_m),
    )

    from src.adapters.imu_to_grf_input import build_grf_input_from_imu_csv
    from src.vnext.data.imu_schema import get_feature_columns

    X = build_grf_input_from_imu_csv(
        fixture_path,
        window_len=256,
        num_windows=64,
        stride=1,
    )
    if X.shape != (64, 256, 12):
        raise ValueError(f"unexpected GRF input tensor shape {X.shape}")

    feature_cols = list(get_feature_columns())
    idx = {n: i for i, n in enumerate(feature_cols)}

    def _col(name: str) -> np.ndarray:
        if name not in idx:
            raise KeyError(f"missing required channel {name}")
        return np.asarray(X[0, :, idx[name]], dtype=np.float32)

    g_thigh = np.sqrt(
        _col("gxx_thigh").astype(np.float64) ** 2
        + _col("gxy_thigh").astype(np.float64) ** 2
        + _col("gxz_thigh").astype(np.float64) ** 2
    ).astype(np.float32, copy=False)
    g_shank = np.sqrt(
        _col("gxx_shank").astype(np.float64) ** 2
        + _col("gxy_shank").astype(np.float64) ** 2
        + _col("gxz_shank").astype(np.float64) ** 2
    ).astype(np.float32, copy=False)

    rel_g = np.abs(g_shank.astype(np.float64) - g_thigh.astype(np.float64)).astype(
        np.float32, copy=False
    )

    window = int(analytics.get("smooth_window_samples"))
    window = max(1, window)
    if window == 1:
        rel_g_sm = rel_g
    else:
        pad_l = window // 2
        pad_r = int(window - 1 - pad_l)
        xp = np.pad(rel_g.astype(np.float64), (pad_l, pad_r), mode="edge")
        kernel = np.ones(int(window), dtype=np.float64) / float(window)
        rel_g_sm = np.convolve(xp, kernel, mode="valid").astype(np.float32, copy=False)

    rel_g_norm = np.clip(rel_g_sm.astype(np.float64) / 2.5, 0.0, 1.0).astype(
        np.float32, copy=False
    )

    lever_arm_base_m = float(args.lever_arm_m)
    lever_arm_gain_m = 0.02
    lever_arm_dyn_m = (
        lever_arm_base_m + float(lever_arm_gain_m) * rel_g_norm.astype(np.float64)
    ).astype(np.float32, copy=False)

    moment_nm_per_kg_raw = (
        (fz_first.astype(np.float64) * lever_arm_dyn_m.astype(np.float64)) / float(args.mass_kg)
    ).astype(np.float32, copy=False)

    if window == 1:
        curve = moment_nm_per_kg_raw
    else:
        pad_l = window // 2
        pad_r = int(window - 1 - pad_l)
        xp = np.pad(moment_nm_per_kg_raw.astype(np.float64), (pad_l, pad_r), mode="edge")
        kernel = np.ones(int(window), dtype=np.float64) / float(window)
        curve = np.convolve(xp, kernel, mode="valid").astype(np.float32, copy=False)

    curve = np.asarray(curve, dtype=np.float32).reshape(-1)
    stats = summarize_curve(curve)
    peak = float(stats["max"])
    p95 = float(stats["p95"])
    finite_fraction = float(stats["finite_fraction"])

    stance = _stance_window_from_fz(
        fz_first,
        body_mass_kg=float(args.mass_kg),
        sample_hz=float(sample_hz),
        smooth_window_samples=int(analytics.get("smooth_window_samples")),
    )

    stance_start_idx = int(stance.get("start_idx", 0))
    stance_end_idx = int(stance.get("end_idx", int(curve.size - 1)))
    if curve.size == 0:
        raise ValueError("empty moment curve")
    stance_start_idx = max(0, min(int(stance_start_idx), int(curve.size - 1)))
    stance_end_idx = max(0, min(int(stance_end_idx), int(curve.size - 1)))
    if stance_end_idx < stance_start_idx:
        stance_start_idx = 0
        stance_end_idx = int(curve.size - 1)

    seg_curve = curve[int(stance_start_idx) : int(stance_end_idx) + 1].astype(np.float64, copy=False)
    smoothness_region_samples = int(seg_curve.size)

    fz_seg = fz_first_n_per_kg[int(stance_start_idx) : int(stance_end_idx) + 1].astype(
        np.float64, copy=False
    )
    if fz_seg.size:
        fz_range_n_per_kg = float(np.percentile(fz_seg, 95.0) - np.percentile(fz_seg, 5.0))
    else:
        fz_range_n_per_kg = 0.0

    if seg_curve.size:
        moment_range_nm_per_kg = float(
            np.percentile(seg_curve, 95.0) - np.percentile(seg_curve, 5.0)
        )
    else:
        moment_range_nm_per_kg = 0.0

    if seg_curve.size >= 2:
        d1 = np.diff(seg_curve)
        abs_d1 = np.abs(d1)
        smoothness = float(np.max(abs_d1))
        smoothness_p95_abs_first_diff = float(np.percentile(abs_d1, 95.0))
    else:
        smoothness = 0.0
        smoothness_p95_abs_first_diff = 0.0

    if seg_curve.size >= 3:
        d2 = np.diff(seg_curve, n=2)
        abs_d2 = np.abs(d2)
        smoothness_max_abs_second_diff = float(np.max(abs_d2))
        smoothness_p95_abs_second_diff = float(np.percentile(abs_d2, 95.0))
    else:
        smoothness_max_abs_second_diff = 0.0
        smoothness_p95_abs_second_diff = 0.0

    peak_fz_n_per_kg = float(np.max(fz_first_n_per_kg.astype(np.float64)))
    peak_moment_nm_per_kg = float(peak)

    sanity = []
    fz_band = (4.0, 20.0)
    moment_band = (0.05, 1.50)
    if not (fz_band[0] <= peak_fz_n_per_kg <= fz_band[1]):
        msg = (
            "WARN sanity: peak_fz_n_per_kg outside plausible walking band "
            f"[{fz_band[0]}, {fz_band[1]}], got {peak_fz_n_per_kg:.6g}"
        )
        print(msg)
        sanity.append({"kind": "peak_fz_n_per_kg", "status": "warn", "message": msg, "band": list(fz_band)})
    else:
        sanity.append({"kind": "peak_fz_n_per_kg", "status": "ok", "band": list(fz_band)})

    if not (moment_band[0] <= peak_moment_nm_per_kg <= moment_band[1]):
        msg = (
            "WARN sanity: peak_moment_nm_per_kg outside plausible walking band "
            f"[{moment_band[0]}, {moment_band[1]}], got {peak_moment_nm_per_kg:.6g}"
        )
        print(msg)
        sanity.append(
            {"kind": "peak_moment_nm_per_kg", "status": "warn", "message": msg, "band": list(moment_band)}
        )
    else:
        sanity.append({"kind": "peak_moment_nm_per_kg", "status": "ok", "band": list(moment_band)})

    fz_range_band = (0.5, 20.0)
    if fz_range_n_per_kg < float(fz_range_band[0]):
        msg = (
            "WARN sanity: fz_range_n_per_kg too small (flatline guardrail) "
            f"min_expected={float(fz_range_band[0])} got {fz_range_n_per_kg:.6g}"
        )
        print(msg)
        sanity.append({"kind": "fz_range_n_per_kg", "status": "warn", "message": msg, "band": list(fz_range_band)})
    else:
        sanity.append({"kind": "fz_range_n_per_kg", "status": "ok", "band": list(fz_range_band)})

    moment_range_band = (0.01, 2.0)
    if moment_range_nm_per_kg < float(moment_range_band[0]):
        msg = (
            "WARN sanity: moment_range_nm_per_kg too small (flatline guardrail) "
            f"min_expected={float(moment_range_band[0])} got {moment_range_nm_per_kg:.6g}"
        )
        print(msg)
        sanity.append(
            {"kind": "moment_range_nm_per_kg", "status": "warn", "message": msg, "band": list(moment_range_band)}
        )
    else:
        sanity.append({"kind": "moment_range_nm_per_kg", "status": "ok", "band": list(moment_range_band)})

    knee_metrics = {
        "schema_version": str(analytics["schema_version"]),
        "fixture": str(fixture_rel),
        "units": {
            "fz_curve": "N",
            "fz_curve_per_kg": "N/kg",
            "knee_moment_curve": "Nm/kg",
            "smoothness_max_abs_first_diff": "Nm/kg per sample",
            "smoothness_p95_abs_first_diff": "Nm/kg per sample",
            "smoothness_max_abs_second_diff": "Nm/kg per sample^2",
            "smoothness_p95_abs_second_diff": "Nm/kg per sample^2",
            "fz_range_n_per_kg": "N/kg",
            "moment_range_nm_per_kg": "Nm/kg",
        },
        "inputs": {
            "mass_kg": float(args.mass_kg),
            "sample_hz": float(sample_hz),
            "dt_s": float(1.0 / float(sample_hz)),
            "lever_arm_m": float(args.lever_arm_m),
            "lever_arm_model": "dynamic_rel_gyro_v1",
            "lever_arm_base_m": float(lever_arm_base_m),
            "lever_arm_gain_m": float(lever_arm_gain_m),
            "window_len": 256,
            "num_windows": 64,
            "stride": 1,
            "filter_kind": str(analytics.get("filter_kind")),
            "smooth_window_samples": int(analytics.get("smooth_window_samples")),
            "smoothness_region": "stance",
            "normalize_debug": {
                "raw_columns": list(debug.raw_columns),
                "canon_columns": list(debug.canon_columns),
                "missing_canon_columns": list(debug.missing_canon_columns),
            },
            "fz_units": fz_prov,
        },
        "stance": stance,
        "curve_len": int(curve.size),
        "fz_curve_len": int(fz_first.size),
        "smoothness_region_samples": int(smoothness_region_samples),
        "curve_stats": stats,
        "peak_nm_per_kg": float(peak),
        "p95_nm_per_kg": float(p95),
        "finite_fraction": float(finite_fraction),
        "smoothness_max_abs_first_diff": float(smoothness),
        "smoothness_p95_abs_first_diff": float(smoothness_p95_abs_first_diff),
        "smoothness_max_abs_second_diff": float(smoothness_max_abs_second_diff),
        "smoothness_p95_abs_second_diff": float(smoothness_p95_abs_second_diff),
        "fz_range_n_per_kg": float(fz_range_n_per_kg),
        "moment_range_nm_per_kg": float(moment_range_nm_per_kg),
        "peak_fz_n_per_kg": float(peak_fz_n_per_kg),
        "sanity_checks": sanity,
    }

    (out_dir / "knee_metrics.json").write_text(
        json.dumps(knee_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _write_curve_csv(out_dir / "knee_moment_curve.csv", curve)
    _write_fz_csv(out_dir / "fz_curve.csv", fz_first, mass_kg=float(args.mass_kg))
    _write_plot(out_dir / "knee_moment_plot.png", curve)

    (out_dir / "README.md").write_text(
        _artifact_readme_text(knee_metrics=knee_metrics),
        encoding="utf-8",
    )

    prov_lines = [
        f"git_sha={gitsha}",
        f"fixture={fixture_rel}",
        f"mass_kg={float(args.mass_kg)}",
        f"sample_hz={float(sample_hz)}",
        f"lever_arm_m={float(args.lever_arm_m)}",
        f"filter_kind={str(analytics.get('filter_kind'))}",
        f"fz_units={str(fz_prov.get('units', 'unknown'))}",
        f"target_norm_path={str(fz_prov.get('target_norm_path', 'unknown'))}",
    ]
    (out_dir / "provenance.txt").write_text("\n".join(prov_lines) + "\n", encoding="utf-8")

    for p in out_dir.rglob("._*"):
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass

    print(f"OUT_DIR={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
