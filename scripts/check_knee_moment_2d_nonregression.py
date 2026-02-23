#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

SCHEMA_VERSION = "knee_moment_2d_contract_v1"
FIXTURE_REL_PATH = "tests/fixtures/imu_sample.csv"
WINDOW_LEN = 256
STRIDE = 128
NUM_WINDOWS = 64

FS_HZ = 200.0
BODY_MASS_KG = 70.0
L_SHANK_M = 0.40

ABS_TOL = 1e-6

FZ_DENORM_RUN_DIR_REL = "data/vnext_gt_real_out/vnext_fz/20260113-161742_0f9d0c7e"

REGEN_BASELINE_CMD = (
    "python scripts/check_knee_moment_2d_nonregression.py "
    "--mode compute --baseline tests/baselines/knee_moment_2d_contract_baseline.json"
)

EXIT_CONTRACT_MISMATCH = 51
EXIT_NONFINITE = 52
EXIT_SHAPE_DTYPE_MISMATCH = 53
EXIT_RADIANS_SANITY = 54
EXIT_MAGNITUDE_BOUNDS = 55
EXIT_NONDETERMINISTIC = 56


class NonFiniteError(Exception):
    def __init__(self, finite_fraction: float) -> None:
        super().__init__("non-finite values")
        self.finite_fraction = float(finite_fraction)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_sys_path() -> None:
    root = _repo_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def _finite_fraction(x: np.ndarray) -> float:
    flat = np.asarray(x).reshape(-1)
    total = int(flat.size)
    if total == 0:
        return 1.0
    return float(np.isfinite(flat).sum()) / float(total)


def _sha256_fingerprint(x: np.ndarray) -> str:
    flat32 = np.asarray(x, dtype=np.float32).reshape(-1)
    if flat32.size:
        payload = ",".join(f"{float(v):.6f}" for v in flat32)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return hashlib.sha256(b"").hexdigest()


def _build_model_and_predict_fz_bw(X: np.ndarray, feature_cols: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
    from src.vnext.data.imu_schema import get_sensor_slices  # type: ignore

    sensor_slices = get_sensor_slices(feature_cols)
    C_canon = int(X.shape[2])

    model_kind = "unknown"

    try:
        from src.vnext.models.vnext_fz import VNextFzModel  # type: ignore

        model_obj: Any = VNextFzModel(in_channels=C_canon, sensor_slices=sensor_slices)
        model_kind = "vnext_fz"
    except Exception:
        model_obj = None

    if model_obj is None:

        class _TinyGRFModel(torch.nn.Module):  # type: ignore[misc]
            def __init__(self, in_channels: int) -> None:
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(in_channels, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 1),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                b, t, c = x.shape
                x_flat = x.reshape(b * t, c)
                y_flat = self.net(x_flat)
                return y_flat.reshape(b, t, 1)

        model_obj = _TinyGRFModel(in_channels=C_canon)
        model_kind = "tiny_fallback"

    model_obj = model_obj.cpu().eval()
    with torch.no_grad():
        y = model_obj(torch.from_numpy(np.asarray(X, dtype=np.float32)).cpu())
    y_np = y.detach().cpu().numpy().astype(np.float32, copy=False)
    return y_np, {"kind": model_kind, "sensor_slices": {k: [int(v.start), int(v.stop)] for k, v in sensor_slices.items()}}


def _compute_outputs(*, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    _ensure_sys_path()

    from src.adapters.imu_to_grf_input import build_grf_input_from_imu_csv  # type: ignore
    from src.vnext.data.imu_schema import get_feature_columns  # type: ignore
    from src.vnext.biomech.complementary_filter import (  # type: ignore
        CANONICAL_VNEXT_IMU_MAPPING,
        ComplementaryFilterConfig,
        complementary_filter_pitch,
        extract_sensor_accel_gyro_from_windows,
    )
    from src.vnext.biomech.fz_units import to_newtons  # type: ignore
    from src.vnext.biomech.knee_moment_2d import KneeMoment2DConfig, estimate_knee_moment_2d  # type: ignore
    from src.vnext.biomech.knee_moment_artifacts import write_knee_moment_2d_artifacts  # type: ignore

    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))

    fixture_path = _repo_root() / FIXTURE_REL_PATH

    X = build_grf_input_from_imu_csv(
        fixture_path,
        window_len=int(WINDOW_LEN),
        stride=int(STRIDE),
        num_windows=int(NUM_WINDOWS),
    )

    feature_cols = get_feature_columns()
    if X.shape != (int(NUM_WINDOWS), int(WINDOW_LEN), int(len(feature_cols))):
        raise SystemExit(
            f"Unexpected adapter shape; got {tuple(X.shape)}, expected "
            f"({int(NUM_WINDOWS)}, {int(WINDOW_LEN)}, {int(len(feature_cols))})"
        )

    if not np.isfinite(X).all():
        raise SystemExit("Non-finite values in GRF input tensor")

    y_fz_bw_btl, grf_model_prov = _build_model_and_predict_fz_bw(X, list(feature_cols))
    if y_fz_bw_btl.shape != (int(NUM_WINDOWS), int(WINDOW_LEN), 1):
        raise SystemExit(f"Unexpected Fz output shape: {tuple(y_fz_bw_btl.shape)}")

    fz_n_bt, fz_prov = to_newtons(
        y_fz_bw_btl[:, :, 0],
        body_mass_kg=float(BODY_MASS_KG),
        g=9.81,
        run_dir=_repo_root() / FZ_DENORM_RUN_DIR_REL,
    )
    fz_n_bt = np.asarray(fz_n_bt, dtype=np.float32)

    a_btc, g_btc = extract_sensor_accel_gyro_from_windows(
        X,
        feature_cols,
        sensor_tag="shank",
        mapping=CANONICAL_VNEXT_IMU_MAPPING,
    )

    theta_bt = np.zeros((int(NUM_WINDOWS), int(WINDOW_LEN)), dtype=np.float64)
    cf_cfg = ComplementaryFilterConfig(alpha=0.98)
    for b in range(int(NUM_WINDOWS)):
        theta_bt[b] = complementary_filter_pitch(a_btc[b], g_btc[b], fs_hz=float(FS_HZ), cfg=cf_cfg)

    km_cfg = KneeMoment2DConfig(
        x_grf_from_ankle_m=0.02,
        enforce_theta_radians=True,
        peak_moment_bounds_nm_per_kg=(-3.0, 3.0),
    )

    res = estimate_knee_moment_2d(
        theta_bt,
        fz_n_bt,
        fs_hz=float(FS_HZ),
        body_mass_kg=float(BODY_MASS_KG),
        l_shank_m=float(L_SHANK_M),
        cfg=km_cfg,
    )

    res.metadata["complementary_filter"] = {
        "alpha": float(cf_cfg.alpha),
        "mapping": {
            "accel_cols": list(CANONICAL_VNEXT_IMU_MAPPING.accel_cols),
            "gyro_cols": list(CANONICAL_VNEXT_IMU_MAPPING.gyro_cols),
        },
        "sensor_tag": "shank",
    }
    res.metadata["imu_to_grf_model"] = dict(grf_model_prov)
    res.metadata["fz_units"] = dict(fz_prov)
    res.metadata["fz_denorm_source"] = str((_repo_root() / FZ_DENORM_RUN_DIR_REL) / "target_norm.json")

    res32 = replace(
        res,
        moment=np.asarray(res.moment, dtype=np.float32),
        moment_filtered=np.asarray(res.moment_filtered, dtype=np.float32),
        theta=np.asarray(res.theta, dtype=np.float32),
        theta_filtered=np.asarray(res.theta_filtered, dtype=np.float32),
        omega=np.asarray(res.omega, dtype=np.float32),
        alpha=np.asarray(res.alpha, dtype=np.float32),
        terms={k: np.asarray(v, dtype=np.float32) for k, v in res.terms.items()},
    )

    out_dir = _repo_root() / "artifacts" / "knee_moment_2d"
    write_knee_moment_2d_artifacts(
        out_dir=out_dir,
        run_id="contract",
        theta_shank_rad=theta_bt,
        fz_n=fz_n_bt,
        result=res32,
        overwrite=True,
    )

    return theta_bt, fz_n_bt, np.asarray(res.moment_filtered, dtype=np.float32)


def _compute_contract() -> Dict[str, Any]:
    theta_bt, fz_n_bt, moment_bt = _compute_outputs(seed=12345)

    theta_max_abs = float(np.nanmax(np.abs(theta_bt)))
    if not np.isfinite(theta_max_abs):
        raise SystemExit(EXIT_RADIANS_SANITY)
    if theta_max_abs > 6.5:
        raise SystemExit(EXIT_RADIANS_SANITY)

    peak_abs = float(np.nanmax(np.abs(moment_bt)))
    if not np.isfinite(peak_abs):
        raise SystemExit(EXIT_MAGNITUDE_BOUNDS)
    if peak_abs < 0.05 or peak_abs > 3.0:
        raise SystemExit(EXIT_MAGNITUDE_BOUNDS)

    finite_fraction = _finite_fraction(moment_bt)
    if finite_fraction != 1.0:
        raise NonFiniteError(finite_fraction)

    flat64 = moment_bt.reshape(-1).astype(np.float64, copy=False)
    global_mean = float(flat64.mean()) if flat64.size else 0.0
    global_std = float(flat64.std()) if flat64.size else 0.0
    mean_abs = float(np.mean(np.abs(flat64))) if flat64.size else 0.0
    p50 = float(np.percentile(flat64, 50.0)) if flat64.size else 0.0
    p95 = float(np.percentile(flat64, 95.0)) if flat64.size else 0.0

    contract: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "fixture": FIXTURE_REL_PATH,
        "window_len": int(WINDOW_LEN),
        "stride": int(STRIDE),
        "num_windows": int(NUM_WINDOWS),
        "fs_hz": float(FS_HZ),
        "body_mass_kg": float(BODY_MASS_KG),
        "l_shank_m": float(L_SHANK_M),
        "theta": {
            "shape": [int(d) for d in theta_bt.shape],
            "dtype": str(np.dtype(theta_bt.dtype)),
            "finite_fraction": float(_finite_fraction(theta_bt)),
            "max_abs": float(theta_max_abs),
            "fingerprint": _sha256_fingerprint(theta_bt),
        },
        "fz_n": {
            "shape": [int(d) for d in fz_n_bt.shape],
            "dtype": str(np.dtype(fz_n_bt.dtype)),
            "finite_fraction": float(_finite_fraction(fz_n_bt)),
            "fingerprint": _sha256_fingerprint(fz_n_bt),
        },
        "moment_filtered": {
            "shape": [int(d) for d in moment_bt.shape],
            "dtype": str(np.dtype(moment_bt.dtype)),
            "finite_fraction": float(finite_fraction),
            "global_mean": float(global_mean),
            "global_std": float(global_std),
            "mean_abs": float(mean_abs),
            "p50": float(p50),
            "p95": float(p95),
            "peak_abs_nm_per_kg": float(peak_abs),
            "output_fingerprint": _sha256_fingerprint(moment_bt),
            "output_flat": [float(v) for v in flat64],
        },
    }

    theta_bt2, fz_n_bt2, moment_bt2 = _compute_outputs(seed=12345)
    if _sha256_fingerprint(theta_bt2) != contract["theta"]["fingerprint"]:
        raise SystemExit(EXIT_NONDETERMINISTIC)
    if _sha256_fingerprint(fz_n_bt2) != contract["fz_n"]["fingerprint"]:
        raise SystemExit(EXIT_NONDETERMINISTIC)
    if _sha256_fingerprint(moment_bt2) != contract["moment_filtered"]["output_fingerprint"]:
        raise SystemExit(EXIT_NONDETERMINISTIC)

    return contract


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"KNEE_MOMENT_2D_CONTRACT missing baseline JSON: {path}")
        raise SystemExit(2)
    data = json.loads(text)
    if not isinstance(data, dict):
        print(f"KNEE_MOMENT_2D_CONTRACT baseline JSON is not an object: {path}")
        raise SystemExit(2)
    return data


def _handle_nonfinite(exc: NonFiniteError, regen_cmd: str) -> int:
    finite_fraction = float(exc.finite_fraction)
    print("FAIL knee_moment_2d_contract: non-finite values")
    print(f"finite_fraction={finite_fraction:.9f}")
    print(f"REGEN_BASELINE_CMD={regen_cmd}")
    return EXIT_NONFINITE


def _run_stats_check(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
    baseline_path: Path,
    *,
    regen_cmd: str,
    print_regen_cmd: bool,
) -> int:
    print(f"KNEE_MOMENT_2D_CONTRACT baseline: {baseline_path}")
    print(f"KNEE_MOMENT_2D_CONTRACT fixture: {FIXTURE_REL_PATH}")

    base = baseline.get("moment_filtered", {})
    curr = current.get("moment_filtered", {})

    base_shape = list(map(int, base.get("shape", [])))
    curr_shape = list(map(int, curr.get("shape", [])))
    base_dtype = str(base.get("dtype"))
    curr_dtype = str(curr.get("dtype"))

    if base_shape != curr_shape or base_dtype != curr_dtype:
        print("FAIL knee_moment_2d_contract: shape/dtype mismatch")
        print(f"baseline_shape={base_shape}")
        print(f"current_shape={curr_shape}")
        print(f"baseline_dtype={base_dtype}")
        print(f"current_dtype={curr_dtype}")
        print(f"REGEN_BASELINE_CMD={regen_cmd}")
        return EXIT_SHAPE_DTYPE_MISMATCH

    base_finite = float(base.get("finite_fraction", 0.0))
    curr_finite = float(curr.get("finite_fraction", 0.0))
    if base_finite != 1.0 or curr_finite != 1.0:
        print("FAIL knee_moment_2d_contract: non-finite values")
        print(f"baseline_finite_fraction={base_finite:.9f}")
        print(f"current_finite_fraction={curr_finite:.9f}")
        print(f"REGEN_BASELINE_CMD={regen_cmd}")
        return EXIT_NONFINITE

    base_fp = str(base.get("output_fingerprint", ""))
    curr_fp = str(curr.get("output_fingerprint", ""))
    if base_fp != curr_fp:
        print("FAIL knee_moment_2d_contract: fingerprint mismatch")
        print(f"FINGERPRINT baseline={base_fp}")
        print(f"FINGERPRINT current={curr_fp}")
        print(f"REGEN_BASELINE_CMD={regen_cmd}")
        return EXIT_CONTRACT_MISMATCH

    base_flat = np.asarray(base.get("output_flat", []), dtype=np.float64)
    curr_flat = np.asarray(curr.get("output_flat", []), dtype=np.float64)
    if base_flat.shape != curr_flat.shape:
        print("FAIL knee_moment_2d_contract: shape mismatch")
        print(f"baseline_output_flat_shape={base_flat.shape}")
        print(f"current_output_flat_shape={curr_flat.shape}")
        print(f"REGEN_BASELINE_CMD={regen_cmd}")
        return EXIT_SHAPE_DTYPE_MISMATCH

    rmse = float(np.sqrt(np.mean((curr_flat - base_flat) ** 2))) if base_flat.size else 0.0

    curr_mean = float(curr.get("global_mean", 0.0))
    curr_std = float(curr.get("global_std", 0.0))
    curr_mean_abs = float(curr.get("mean_abs", 0.0))
    curr_p50 = float(curr.get("p50", 0.0))
    curr_p95 = float(curr.get("p95", 0.0))
    curr_peak = float(curr.get("peak_abs_nm_per_kg", 0.0))

    print(
        "KNEE_MOMENT_2D_CONTRACT current: "
        f"shape={curr_shape} "
        f"dtype={curr_dtype} "
        f"finite_fraction={curr_finite:.9f} "
        f"global_mean={curr_mean:.6g} "
        f"global_std={curr_std:.6g} "
        f"mean_abs={curr_mean_abs:.6g} "
        f"p50={curr_p50:.6g} "
        f"p95={curr_p95:.6g} "
        f"peak_abs_nm_per_kg={curr_peak:.6g} "
        f"global_rmse={rmse:.6g}"
    )

    if not np.isfinite(curr_peak) or curr_peak > 3.0:
        print("FAIL knee_moment_2d_contract: magnitude bounds")
        print(f"current_peak_abs_nm_per_kg={curr_peak:.9g}")
        print(f"REGEN_BASELINE_CMD={regen_cmd}")
        return EXIT_MAGNITUDE_BOUNDS

    if curr_peak < 0.05:
        print("FAIL knee_moment_2d_contract: magnitude bounds")
        print(f"current_peak_abs_nm_per_kg={curr_peak:.9g}")
        print(f"REGEN_BASELINE_CMD={regen_cmd}")
        return EXIT_MAGNITUDE_BOUNDS

    drift_records = []

    def _check_stat(name: str, base_val: float, curr_val: float) -> None:
        abs_diff = abs(curr_val - base_val)
        if abs_diff > ABS_TOL:
            drift_records.append((name, base_val, curr_val, abs_diff))

    _check_stat("global_mean", float(base.get("global_mean", 0.0)), curr_mean)
    _check_stat("global_std", float(base.get("global_std", 0.0)), curr_std)
    _check_stat("mean_abs", float(base.get("mean_abs", 0.0)), curr_mean_abs)
    _check_stat("p50", float(base.get("p50", 0.0)), curr_p50)
    _check_stat("p95", float(base.get("p95", 0.0)), curr_p95)
    _check_stat("peak_abs_nm_per_kg", float(base.get("peak_abs_nm_per_kg", 0.0)), curr_peak)

    if drift_records:
        print("FAIL knee_moment_2d_contract: stats drift")
        for name, base_val, curr_val, abs_diff in drift_records:
            print(
                f"DRIFT {name} "
                f"baseline={base_val:.9g} "
                f"current={curr_val:.9g} "
                f"abs_diff={abs_diff:.3g} "
                f"tol={ABS_TOL}"
            )
        print(f"REGEN_BASELINE_CMD={regen_cmd}")
        return EXIT_CONTRACT_MISMATCH

    print("PASS knee_moment_2d_contract")
    if print_regen_cmd:
        print(f"REGEN_BASELINE_CMD={regen_cmd}")
    return 0


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute or check the knee moment 2D non-regression contract on the canonical IMU sample fixture."
        )
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Path to baseline contract JSON (for both compute and check modes)",
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["compute", "check"],
        help="Mode: compute baseline JSON or check against it",
    )
    parser.add_argument(
        "--print_regen_cmd",
        action="store_true",
        help=(
            "In check mode, print a helper command for regenerating the baseline JSON (does not modify files)."
        ),
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = _parse_args(argv)
    baseline_path = Path(args.baseline)

    regen_cmd = REGEN_BASELINE_CMD

    if args.mode == "compute":
        contract = _compute_contract()
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    baseline = _load_json(baseline_path)

    try:
        current = _compute_contract()
    except NonFiniteError as exc:
        raise SystemExit(_handle_nonfinite(exc, regen_cmd))

    code = _run_stats_check(
        baseline,
        current,
        baseline_path,
        regen_cmd=regen_cmd,
        print_regen_cmd=getattr(args, "print_regen_cmd", False),
    )
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main(sys.argv[1:])
