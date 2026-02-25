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

    analytics = compute_knee_moment_from_fz(
        fz_first,
        mass_kg=float(args.mass_kg),
        sample_hz=float(sample_hz),
        stride_len_m=float(args.lever_arm_m),
    )

    curve = np.asarray(analytics["moment_nm_per_kg"], dtype=np.float32).reshape(-1)
    stats = summarize_curve(curve)
    peak = float(stats["max"])
    p95 = float(stats["p95"])
    finite_fraction = float(stats["finite_fraction"])

    if curve.size >= 2:
        diffs = np.diff(curve.astype(np.float64))
        smoothness = float(np.max(np.abs(diffs)))
    else:
        smoothness = 0.0

    knee_metrics = {
        "schema_version": str(analytics["schema_version"]),
        "fixture": str(fixture_rel),
        "inputs": {
            "mass_kg": float(args.mass_kg),
            "sample_hz": float(sample_hz),
            "lever_arm_m": float(args.lever_arm_m),
            "window_len": 256,
            "num_windows": 64,
            "stride": 1,
            "filter_kind": str(analytics.get("filter_kind")),
            "normalize_debug": {
                "raw_columns": list(debug.raw_columns),
                "canon_columns": list(debug.canon_columns),
                "missing_canon_columns": list(debug.missing_canon_columns),
            },
            "fz_units": fz_prov,
        },
        "curve_len": int(curve.size),
        "curve_stats": stats,
        "peak_nm_per_kg": float(peak),
        "p95_nm_per_kg": float(p95),
        "finite_fraction": float(finite_fraction),
        "smoothness_max_abs_first_diff": float(smoothness),
    }

    (out_dir / "knee_metrics.json").write_text(
        json.dumps(knee_metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    _write_curve_csv(out_dir / "knee_moment_curve.csv", curve)
    _write_plot(out_dir / "knee_moment_plot.png", curve)

    prov_lines = [
        f"git_sha={gitsha}",
        f"fixture={fixture_rel}",
        f"mass_kg={float(args.mass_kg)}",
        f"sample_hz={float(sample_hz)}",
        f"lever_arm_m={float(args.lever_arm_m)}",
        f"filter_kind={str(analytics.get('filter_kind'))}",
    ]
    (out_dir / "provenance.txt").write_text("\n".join(prov_lines) + "\n", encoding="utf-8")

    print(f"OUT_DIR={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
