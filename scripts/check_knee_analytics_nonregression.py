#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

SCHEMA_VERSION = "knee_analytics_walk_contract_v1"
FIXTURE_REL_PATH = "tests/fixtures/imu_sample.csv"

EXIT_CONTRACT_MISMATCH = 61
EXIT_INVALID_OUTPUT = 62

REGEN_BASELINE_CMD = (
    "python3 scripts/check_knee_analytics_nonregression.py "
    "--mode compute --baseline tests/baselines/knee_analytics_walk_contract_baseline.json"
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_sys_path() -> None:
    root = _repo_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def _load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("json root must be object")
    return obj


def _read_curve_csv(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8").splitlines()
    if not text or not text[0].strip().startswith("idx,moment_nm_per_kg"):
        raise ValueError("bad curve csv header")
    vals = []
    for line in text[1:]:
        if not line.strip():
            continue
        parts = line.split(",")
        if len(parts) != 2:
            raise ValueError("bad curve csv row")
        vals.append(float(parts[1]))
    return np.asarray(vals, dtype=np.float32)


def _compute_current_from_out_dir(out_dir: Path) -> Dict[str, Any]:
    metrics_path = out_dir / "knee_metrics.json"
    curve_path = out_dir / "knee_moment_curve.csv"
    if not metrics_path.exists() or not curve_path.exists():
        raise FileNotFoundError("missing required artifacts")

    metrics = _load_json(metrics_path)
    curve = _read_curve_csv(curve_path)

    if int(curve.size) != 256:
        raise ValueError("unexpected curve length")
    if not np.isfinite(curve).all():
        raise ValueError("non-finite curve")

    stance_obj = metrics.get("stance")
    if not isinstance(stance_obj, dict):
        raise ValueError("missing stance")
    stance_start_idx = int(stance_obj.get("start_idx", 0))
    stance_end_idx = int(stance_obj.get("end_idx", int(curve.size - 1)))
    stance_start_idx = max(0, min(int(stance_start_idx), int(curve.size - 1)))
    stance_end_idx = max(0, min(int(stance_end_idx), int(curve.size - 1)))
    if stance_end_idx < stance_start_idx:
        stance_start_idx = 0
        stance_end_idx = int(curve.size - 1)

    seg_curve = curve[int(stance_start_idx) : int(stance_end_idx) + 1].astype(np.float64, copy=False)
    smoothness_region_samples = int(seg_curve.size)

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

    peak = float(np.max(curve.astype(np.float64)))
    p95 = float(np.percentile(curve.astype(np.float64), 95.0))

    required_keys = [
        "schema_version",
        "fixture",
        "inputs",
        "inputs.smoothness_region",
        "inputs.lever_arm_model",
        "inputs.lever_arm_base_m",
        "inputs.lever_arm_gain_m",
        "curve_len",
        "curve_stats",
        "peak_nm_per_kg",
        "p95_nm_per_kg",
        "finite_fraction",
        "smoothness_max_abs_first_diff",
        "smoothness_p95_abs_first_diff",
        "smoothness_max_abs_second_diff",
        "smoothness_p95_abs_second_diff",
        "smoothness_region_samples",
        "fz_range_n_per_kg",
        "moment_range_nm_per_kg",
    ]

    for k in required_keys:
        cur: Any = metrics
        for part in str(k).split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise ValueError(f"missing key {k}")
            cur = cur[part]

    if str(metrics.get("schema_version")) != "knee_analytics_v1":
        raise ValueError("unexpected knee_metrics schema_version")
    if str(metrics.get("fixture")) != FIXTURE_REL_PATH:
        raise ValueError("unexpected fixture")

    smoothness_region = ((metrics.get("inputs") or {}).get("smoothness_region"))
    if str(smoothness_region) != "stance":
        raise ValueError("inputs.smoothness_region must be 'stance'")

    lever_arm_model = ((metrics.get("inputs") or {}).get("lever_arm_model"))
    if str(lever_arm_model) != "dynamic_rel_gyro_v1":
        raise ValueError("inputs.lever_arm_model must be 'dynamic_rel_gyro_v1'")

    finite_fraction = float(metrics.get("finite_fraction", 0.0))
    if finite_fraction != 1.0:
        raise ValueError("finite_fraction must be 1.0")

    fz_range_n_per_kg = float(metrics.get("fz_range_n_per_kg", 0.0))
    moment_range_nm_per_kg = float(metrics.get("moment_range_nm_per_kg", 0.0))

    return {
        "schema_version": SCHEMA_VERSION,
        "fixture": FIXTURE_REL_PATH,
        "required_keys": required_keys,
        "curve_len": int(curve.size),
        "peak_nm_per_kg": float(peak),
        "p95_nm_per_kg": float(p95),
        "smoothness_max_abs_first_diff": float(smoothness),
        "smoothness_p95_abs_first_diff": float(smoothness_p95_abs_first_diff),
        "smoothness_max_abs_second_diff": float(smoothness_max_abs_second_diff),
        "smoothness_p95_abs_second_diff": float(smoothness_p95_abs_second_diff),
        "smoothness_region_samples": int(smoothness_region_samples),
        "fz_range_n_per_kg": float(fz_range_n_per_kg),
        "moment_range_nm_per_kg": float(moment_range_nm_per_kg),
    }


def _parse_out_dir_from_runner_output(text: str) -> Path:
    out_dir = None
    for line in text.splitlines():
        m = re.match(r"^OUT_DIR=(.+)$", line.strip())
        if m:
            out_dir = m.group(1).strip()
    if out_dir is None:
        raise ValueError("runner did not print OUT_DIR")
    return Path(out_dir)


def _run_runner() -> Path:
    cmd = ["python3", "scripts/run_knee_analytics_walking.py"]
    proc = subprocess.run(
        cmd,
        cwd=str(_repo_root()),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        sys.stdout.write(proc.stdout)
        raise SystemExit(proc.returncode)
    out_dir = _parse_out_dir_from_runner_output(proc.stdout)
    return out_dir


def _make_band(x: float, frac: float, abs_pad: float) -> Tuple[float, float]:
    lo = x * (1.0 - frac) - abs_pad
    hi = x * (1.0 + frac) + abs_pad
    return float(lo), float(hi)


def _compare(baseline: Dict[str, Any], current: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    ok = True

    if baseline.get("schema_version") != current.get("schema_version"):
        ok = False
    if baseline.get("fixture") != current.get("fixture"):
        ok = False
    if list(baseline.get("required_keys", [])) != list(current.get("required_keys", [])):
        ok = False
    if int(baseline.get("curve_len", -1)) != int(current.get("curve_len", -1)):
        ok = False

    peak_band = baseline.get("peak_nm_per_kg_band")
    p95_band = baseline.get("p95_nm_per_kg_band")
    smooth_max = float(baseline.get("smoothness_max_abs_first_diff_max", 0.0))
    baseline_smooth_p95_d1 = float(baseline.get("smoothness_p95_abs_first_diff", 0.0))
    baseline_smooth_max_d2 = float(baseline.get("smoothness_max_abs_second_diff", 0.0))
    baseline_smooth_p95_d2 = float(baseline.get("smoothness_p95_abs_second_diff", 0.0))

    peak = float(current.get("peak_nm_per_kg", 0.0))
    p95 = float(current.get("p95_nm_per_kg", 0.0))
    smooth = float(current.get("smoothness_max_abs_first_diff", 0.0))
    smooth_p95_d1 = float(current.get("smoothness_p95_abs_first_diff", 0.0))
    smooth_max_d2 = float(current.get("smoothness_max_abs_second_diff", 0.0))
    smooth_p95_d2 = float(current.get("smoothness_p95_abs_second_diff", 0.0))
    smoothness_region_samples = int(current.get("smoothness_region_samples", -1))
    fz_range_n_per_kg = float(current.get("fz_range_n_per_kg", 0.0))
    moment_range_nm_per_kg = float(current.get("moment_range_nm_per_kg", 0.0))

    if not (isinstance(peak_band, list) and len(peak_band) == 2):
        ok = False
        peak_ok = False
        peak_lo = float("nan")
        peak_hi = float("nan")
    else:
        peak_lo = float(peak_band[0])
        peak_hi = float(peak_band[1])
        peak_ok = (peak_lo <= peak <= peak_hi) and np.isfinite(peak)

    if not (isinstance(p95_band, list) and len(p95_band) == 2):
        ok = False
        p95_ok = False
        p95_lo = float("nan")
        p95_hi = float("nan")
    else:
        p95_lo = float(p95_band[0])
        p95_hi = float(p95_band[1])
        p95_ok = (p95_lo <= p95 <= p95_hi) and np.isfinite(p95)

    smooth_ok = (smooth <= smooth_max) and np.isfinite(smooth)

    smooth_p95_d1_max = float(max(baseline_smooth_p95_d1 * 1.10, 1e-6))
    smooth_max_d2_max = float(max(baseline_smooth_max_d2 * 1.15, 1e-5))
    smooth_p95_d2_max = float(max(baseline_smooth_p95_d2 * 1.15, 1e-5))

    smooth_p95_ok = (smooth_p95_d1 <= smooth_p95_d1_max) and np.isfinite(smooth_p95_d1)
    smooth_d2_ok = (smooth_max_d2 <= smooth_max_d2_max) and np.isfinite(smooth_max_d2)
    smooth_p95_d2_ok = (smooth_p95_d2 <= smooth_p95_d2_max) and np.isfinite(smooth_p95_d2)

    fz_range_floor = 0.5
    moment_range_floor = 0.01
    fz_range_ok = (fz_range_n_per_kg >= float(fz_range_floor)) and np.isfinite(fz_range_n_per_kg)
    moment_range_ok = (moment_range_nm_per_kg >= float(moment_range_floor)) and np.isfinite(moment_range_nm_per_kg)

    if (
        not peak_ok
        or not p95_ok
        or not smooth_ok
        or not smooth_p95_ok
        or not smooth_d2_ok
        or not smooth_p95_d2_ok
        or smoothness_region_samples <= 0
        or not fz_range_ok
        or not moment_range_ok
    ):
        ok = False

    report = {
        "peak_nm_per_kg": peak,
        "peak_band": [peak_lo, peak_hi],
        "p95_nm_per_kg": p95,
        "p95_band": [p95_lo, p95_hi],
        "smoothness_max_abs_first_diff": smooth,
        "smoothness_max_abs_first_diff_max": smooth_max,
        "smoothness_p95_abs_first_diff": smooth_p95_d1,
        "smoothness_p95_abs_first_diff_baseline": baseline_smooth_p95_d1,
        "smoothness_p95_abs_first_diff_max": float(smooth_p95_d1_max),
        "smoothness_max_abs_second_diff": smooth_max_d2,
        "smoothness_max_abs_second_diff_baseline": baseline_smooth_max_d2,
        "smoothness_max_abs_second_diff_max": float(smooth_max_d2_max),
        "smoothness_p95_abs_second_diff": smooth_p95_d2,
        "smoothness_p95_abs_second_diff_baseline": baseline_smooth_p95_d2,
        "smoothness_p95_abs_second_diff_max": float(smooth_p95_d2_max),
        "smoothness_region_samples": int(smoothness_region_samples),
        "fz_range_n_per_kg": float(fz_range_n_per_kg),
        "fz_range_floor": float(fz_range_floor),
        "moment_range_nm_per_kg": float(moment_range_nm_per_kg),
        "moment_range_floor": float(moment_range_floor),
    }

    return ok, report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["compute", "check"], required=True)
    p.add_argument("--baseline", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="")
    p.add_argument("--print_regen_cmd", action="store_true")
    args = p.parse_args()

    _ensure_sys_path()

    baseline_path = Path(args.baseline)

    if args.mode == "compute":
        out_dir = Path(args.out_dir) if args.out_dir else _run_runner()
        current = _compute_current_from_out_dir(out_dir)

        peak = float(current["peak_nm_per_kg"])
        p95 = float(current["p95_nm_per_kg"])
        smooth = float(current["smoothness_max_abs_first_diff"])
        smooth_p95_d1 = float(current["smoothness_p95_abs_first_diff"])
        smooth_max_d2 = float(current["smoothness_max_abs_second_diff"])
        smooth_p95_d2 = float(current["smoothness_p95_abs_second_diff"])
        fz_range_n_per_kg = float(current["fz_range_n_per_kg"])
        moment_range_nm_per_kg = float(current["moment_range_nm_per_kg"])

        peak_lo, peak_hi = _make_band(peak, frac=0.20, abs_pad=1e-6)
        p95_lo, p95_hi = _make_band(p95, frac=0.20, abs_pad=1e-6)
        smooth_max = float(smooth * 1.50 + 1e-6)

        baseline = {
            "schema_version": SCHEMA_VERSION,
            "fixture": FIXTURE_REL_PATH,
            "required_keys": list(current["required_keys"]),
            "curve_len": int(current["curve_len"]),
            "peak_nm_per_kg_band": [peak_lo, peak_hi],
            "p95_nm_per_kg_band": [p95_lo, p95_hi],
            "smoothness_max_abs_first_diff_max": float(smooth_max),
            "smoothness_p95_abs_first_diff": float(smooth_p95_d1),
            "smoothness_max_abs_second_diff": float(smooth_max_d2),
            "smoothness_p95_abs_second_diff": float(smooth_p95_d2),
            "fz_range_n_per_kg": float(fz_range_n_per_kg),
            "moment_range_nm_per_kg": float(moment_range_nm_per_kg),
        }

        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(baseline, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        print(f"KNEE_ANALYTICS_CONTRACT baseline_written: {baseline_path}")
        if args.print_regen_cmd:
            print(f"REGEN_BASELINE_CMD={REGEN_BASELINE_CMD}")
        return 0

    out_dir = Path(args.out_dir)
    baseline = _load_json(baseline_path)

    try:
        current = _compute_current_from_out_dir(out_dir)
    except Exception as e:
        print(f"FAIL knee_analytics_walk_contract: invalid output in out_dir={out_dir}")
        print(f"ERROR={type(e).__name__}: {e}")
        print(f"REGEN_BASELINE_CMD={REGEN_BASELINE_CMD}")
        return EXIT_INVALID_OUTPUT

    ok, report = _compare(baseline, current)

    print(f"KNEE_ANALYTICS_CONTRACT baseline: {baseline_path}")
    print(f"KNEE_ANALYTICS_CONTRACT fixture: {FIXTURE_REL_PATH}")
    print(
        "KNEE_ANALYTICS_CONTRACT current: "
        f"curve_len={int(current.get('curve_len', -1))} "
        f"peak_nm_per_kg={float(report['peak_nm_per_kg']):.6g} "
        f"p95_nm_per_kg={float(report['p95_nm_per_kg']):.6g} "
        f"smoothness_max_abs_first_diff={float(report['smoothness_max_abs_first_diff']):.6g} "
        f"smoothness_p95_abs_first_diff={float(report['smoothness_p95_abs_first_diff']):.6g} "
        f"smoothness_max_abs_second_diff={float(report['smoothness_max_abs_second_diff']):.6g} "
        f"smoothness_p95_abs_second_diff={float(report['smoothness_p95_abs_second_diff']):.6g} "
        f"smoothness_region_samples={int(report['smoothness_region_samples'])} "
        f"fz_range_n_per_kg={float(report['fz_range_n_per_kg']):.6g} "
        f"moment_range_nm_per_kg={float(report['moment_range_nm_per_kg']):.6g}"
    )

    if not ok:
        print("FAIL knee_analytics_walk_contract")
        print(
            "KNEE_ANALYTICS_CONTRACT limits: "
            f"smooth_p95_d1<={float(report['smoothness_p95_abs_first_diff_max']):.6g} "
            f"smooth_max_d2<={float(report['smoothness_max_abs_second_diff_max']):.6g} "
            f"smooth_p95_d2<={float(report['smoothness_p95_abs_second_diff_max']):.6g} "
            f"fz_range>={float(report['fz_range_floor']):.6g} "
            f"moment_range>={float(report['moment_range_floor']):.6g}"
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        print(f"REGEN_BASELINE_CMD={REGEN_BASELINE_CMD}")
        return EXIT_CONTRACT_MISMATCH

    print("PASS knee_analytics_walk_contract")
    if args.print_regen_cmd:
        print(f"REGEN_BASELINE_CMD={REGEN_BASELINE_CMD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
