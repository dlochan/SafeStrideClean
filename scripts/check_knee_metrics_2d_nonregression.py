#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

SCHEMA_VERSION = "knee_metrics_2d_contract_v1"
FIXTURE_REL_PATH = "tests/fixtures/imu_sample.csv"
WINDOW_LEN = 256
STRIDE = 128
NUM_WINDOWS = 64

FS_HZ = 200.0
BODY_MASS_KG = 70.0
L_SHANK_M = 0.40

FZ_DENORM_RUN_DIR_REL = "data/vnext_gt_real_out/vnext_fz/20260113-161742_0f9d0c7e"

REGEN_BASELINE_CMD = (
    "python3 scripts/check_knee_metrics_2d_nonregression.py "
    "--mode compute --baseline tests/baselines/knee_metrics_2d_contract_baseline.json"
)

EXIT_CONTRACT_MISMATCH = 61
EXIT_NONFINITE = 62
EXIT_SHAPE_DTYPE_MISMATCH = 63
EXIT_MAGNITUDE_BOUNDS = 64
EXIT_NONDETERMINISTIC = 65


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
    from src.vnext.data.imu_schema import get_sensor_slices

    sensor_slices = get_sensor_slices(feature_cols)
    C_canon = int(X.shape[2])

    model_kind = "unknown"

    try:
        from src.vnext.models.vnext_fz import VNextFzModel

        model_obj: Any = VNextFzModel(in_channels=C_canon, sensor_slices=sensor_slices)
        model_kind = "vnext_fz"
    except Exception:
        model_obj = None

    if model_obj is None:

        class _TinyGRFModel(torch.nn.Module):
            def __init__(self, in_channels: int) -> None:
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(in_channels, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 1),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    prov = {
        "kind": model_kind,
        "sensor_slices": {k: [int(v.start), int(v.stop)] for k, v in sensor_slices.items()},
    }
    return y_np, prov


def _compute_metrics(*, seed: int) -> Tuple[np.ndarray, Dict[str, Any], Dict[str, Any]]:
    _ensure_sys_path()

    from src.adapters.imu_to_grf_input import build_grf_input_from_imu_csv
    from src.vnext.data.imu_schema import get_feature_columns
    from src.vnext.biomech.complementary_filter import (
        CANONICAL_VNEXT_IMU_MAPPING,
        ComplementaryFilterConfig,
        complementary_filter_pitch,
        extract_sensor_accel_gyro_from_windows,
    )
    from src.vnext.biomech.fz_units import to_newtons
    from src.vnext.biomech.knee_moment_2d import KneeMoment2DConfig, estimate_knee_moment_2d
    from src.vnext.biomech.knee_metrics_2d import KneeStanceDetection2DConfig, compute_knee_metrics_2d
    from src.vnext.biomech.knee_metrics_artifacts import write_knee_metrics_2d_artifacts

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

    moment_bt = np.asarray(res.moment_filtered, dtype=np.float32)

    stance_cfg = KneeStanceDetection2DConfig(fz_threshold_n=50.0, min_stance_duration_s=0.20)
    per_window, summary = compute_knee_metrics_2d(
        fz_n_bt,
        moment_bt,
        fs_hz=float(FS_HZ),
        cfg=stance_cfg,
    )

    out_dir = _repo_root() / "artifacts" / "knee_metrics_2d"
    write_knee_metrics_2d_artifacts(
        out_dir=out_dir,
        trial_id="contract",
        metrics_per_window=per_window,
        metrics_summary=summary,
        overwrite=True,
    )

    keys = [
        "peak_fz_n_per_kg",
        "impulse_fz_ns_per_kg",
        "loading_rate_n_per_kg_s",
        "peak_knee_moment_nm_per_kg",
        "moment_impulse_nms_per_kg",
        "time_to_peak_moment_s",
    ]

    M = np.zeros((len(per_window), len(keys)), dtype=np.float32)
    for i, r in enumerate(sorted(per_window, key=lambda d: int(d["window_index"]))):
        for j, k in enumerate(keys):
            M[i, j] = np.float32(float(r[k]))

    prov: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "fixture": FIXTURE_REL_PATH,
        "window_len": int(WINDOW_LEN),
        "stride": int(STRIDE),
        "num_windows": int(NUM_WINDOWS),
        "fs_hz": float(FS_HZ),
        "fz_denorm_run_dir": str(_repo_root() / FZ_DENORM_RUN_DIR_REL),
        "imu_to_grf_model": dict(grf_model_prov),
        "fz_units": dict(fz_prov),
        "complementary_filter": {
            "alpha": float(cf_cfg.alpha),
            "sensor_tag": "shank",
        },
    }

    return M, summary, prov


def _compute_contract() -> Dict[str, Any]:
    M, summary, prov = _compute_metrics(seed=12345)

    finite_fraction = _finite_fraction(M)
    if finite_fraction != 1.0:
        raise NonFiniteError(finite_fraction)

    fp = _sha256_fingerprint(M)

    median_peak_fz = float(summary["peak_fz_n_per_kg"]["median"])
    median_peak_moment = float(summary["peak_knee_moment_nm_per_kg"]["median"])

    if not np.isfinite(median_peak_fz) or median_peak_fz <= 3.0:
        raise SystemExit(EXIT_MAGNITUDE_BOUNDS)
    if not np.isfinite(median_peak_moment) or median_peak_moment <= 0.05:
        raise SystemExit(EXIT_MAGNITUDE_BOUNDS)

    contract: Dict[str, Any] = {
        **prov,
        "metrics_matrix": {
            "shape": [int(d) for d in M.shape],
            "dtype": str(np.dtype(M.dtype)),
            "finite_fraction": float(finite_fraction),
            "fingerprint": str(fp),
        },
        "summary": summary,
        "gates": {
            "median_peak_fz_n_per_kg_gt": 3.0,
            "median_peak_knee_moment_nm_per_kg_gt": 0.05,
            "median_peak_fz_n_per_kg": float(median_peak_fz),
            "median_peak_knee_moment_nm_per_kg": float(median_peak_moment),
        },
    }

    M2, summary2, _prov2 = _compute_metrics(seed=12345)
    if _sha256_fingerprint(M2) != fp:
        raise SystemExit(EXIT_NONDETERMINISTIC)

    if float(summary2["peak_fz_n_per_kg"]["median"]) != median_peak_fz:
        raise SystemExit(EXIT_NONDETERMINISTIC)
    if float(summary2["peak_knee_moment_nm_per_kg"]["median"]) != median_peak_moment:
        raise SystemExit(EXIT_NONDETERMINISTIC)

    return contract


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"KNEE_METRICS_2D_CONTRACT missing baseline JSON: {path}")
        raise SystemExit(2)
    data = json.loads(text)
    if not isinstance(data, dict):
        print(f"KNEE_METRICS_2D_CONTRACT baseline JSON is not an object: {path}")
        raise SystemExit(2)
    return data


def _handle_nonfinite(exc: NonFiniteError, regen_cmd: str) -> int:
    finite_fraction = float(exc.finite_fraction)
    print("FAIL knee_metrics_2d_contract: non-finite values")
    print(f"finite_fraction={finite_fraction:.9f}")
    print(f"REGEN_BASELINE_CMD={regen_cmd}")
    return EXIT_NONFINITE


def _run_check(
    baseline: Dict[str, Any],
    current: Dict[str, Any],
    baseline_path: Path,
    *,
    regen_cmd: str,
    print_regen_cmd: bool,
) -> int:
    print(f"KNEE_METRICS_2D_CONTRACT baseline: {baseline_path}")
    print(f"KNEE_METRICS_2D_CONTRACT fixture: {FIXTURE_REL_PATH}")

    base_m = baseline.get("metrics_matrix", {})
    curr_m = current.get("metrics_matrix", {})

    base_shape = list(map(int, base_m.get("shape", [])))
    curr_shape = list(map(int, curr_m.get("shape", [])))
    base_dtype = str(base_m.get("dtype"))
    curr_dtype = str(curr_m.get("dtype"))

    if base_shape != curr_shape or base_dtype != curr_dtype:
        print("FAIL knee_metrics_2d_contract: shape/dtype mismatch")
        print(f"baseline_shape={base_shape}")
        print(f"current_shape={curr_shape}")
        print(f"baseline_dtype={base_dtype}")
        print(f"current_dtype={curr_dtype}")
        print(f"REGEN_BASELINE_CMD={regen_cmd}")
        return EXIT_SHAPE_DTYPE_MISMATCH

    base_finite = float(base_m.get("finite_fraction", 0.0))
    curr_finite = float(curr_m.get("finite_fraction", 0.0))
    if base_finite != 1.0 or curr_finite != 1.0:
        print("FAIL knee_metrics_2d_contract: non-finite values")
        print(f"baseline_finite_fraction={base_finite:.9f}")
        print(f"current_finite_fraction={curr_finite:.9f}")
        print(f"REGEN_BASELINE_CMD={regen_cmd}")
        return EXIT_NONFINITE

    base_fp = str(base_m.get("fingerprint", ""))
    curr_fp = str(curr_m.get("fingerprint", ""))

    med_peak_fz = float(current.get("summary", {}).get("peak_fz_n_per_kg", {}).get("median", 0.0))
    med_peak_m = float(
        current.get("summary", {}).get("peak_knee_moment_nm_per_kg", {}).get("median", 0.0)
    )

    print(
        "KNEE_METRICS_2D_CONTRACT current: "
        f"shape={curr_shape} "
        f"dtype={curr_dtype} "
        f"finite_fraction={curr_finite:.9f} "
        f"median_peak_fz_n_per_kg={med_peak_fz:.6g} "
        f"median_peak_moment_nm_per_kg={med_peak_m:.6g}"
    )

    if not np.isfinite(med_peak_fz) or med_peak_fz <= 3.0:
        print("FAIL knee_metrics_2d_contract: magnitude bounds")
        print(f"median_peak_fz_n_per_kg={med_peak_fz:.9g}")
        print(f"REGEN_BASELINE_CMD={regen_cmd}")
        return EXIT_MAGNITUDE_BOUNDS

    if not np.isfinite(med_peak_m) or med_peak_m <= 0.05:
        print("FAIL knee_metrics_2d_contract: magnitude bounds")
        print(f"median_peak_knee_moment_nm_per_kg={med_peak_m:.9g}")
        print(f"REGEN_BASELINE_CMD={regen_cmd}")
        return EXIT_MAGNITUDE_BOUNDS

    if base_fp != curr_fp:
        print("FAIL knee_metrics_2d_contract: fingerprint mismatch")
        print(f"FINGERPRINT baseline={base_fp}")
        print(f"FINGERPRINT current={curr_fp}")
        print(f"REGEN_BASELINE_CMD={regen_cmd}")
        return EXIT_CONTRACT_MISMATCH

    print("PASS knee_metrics_2d_contract")
    if print_regen_cmd:
        print(f"REGEN_BASELINE_CMD={regen_cmd}")

    return 0


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute or check the knee metrics 2D non-regression contract on the canonical IMU sample fixture."
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

    code = _run_check(
        baseline,
        current,
        baseline_path,
        regen_cmd=regen_cmd,
        print_regen_cmd=getattr(args, "print_regen_cmd", False),
    )
    if code != 0:
        raise SystemExit(code)


if __name__ == "__main__":
    main(sys.argv[1:])
