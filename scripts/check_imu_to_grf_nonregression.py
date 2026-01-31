#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import hashlib
import numpy as np
import torch
import random

SCHEMA_VERSION = "imu_to_grf_contract_v2"
FIXTURE_REL_PATH = "tests/fixtures/imu_sample.csv"
WINDOW_LEN = 256
STRIDE = 1
NUM_WINDOWS = 64

ABS_TOL = 1e-6

REGEN_BASELINE_CMD = (
    "python scripts/check_imu_to_grf_nonregression.py "
    "--mode compute --baseline tests/baselines/imu_to_grf_contract_baseline.json"
)

EXIT_CONTRACT_MISMATCH = 41
EXIT_NONFINITE = 42
EXIT_SHAPE_DTYPE_MISMATCH = 43


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


def _run_inference() -> np.ndarray:
    """Run deterministic IMU→GRF inference and return the output tensor as np.ndarray.

    This mirrors the deterministic path used by the existing IMU→GRF inference
    utilities: fixed seeds, canonical fixture, and a vNext model when
    available with a tiny deterministic fallback otherwise.
    """

    _ensure_sys_path()

    from src.adapters.imu_to_grf_input import build_grf_input_from_imu_csv  # type: ignore
    from src.vnext.data.imu_schema import get_feature_columns, get_sensor_slices  # type: ignore

    try:  # Prefer the real vNext model when available.
        from src.vnext.models.vnext_fz import VNextFzModel  # type: ignore

        have_vnext = True
    except Exception:  # pragma: no cover - defensive
        VNextFzModel = None  # type: ignore
        have_vnext = False

    seed = 12345
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    fixture_path = _repo_root() / FIXTURE_REL_PATH

    X = build_grf_input_from_imu_csv(
        fixture_path,
        window_len=int(WINDOW_LEN),
        num_windows=int(NUM_WINDOWS),
        stride=int(STRIDE),
    )

    if X.dtype != np.float32:
        raise SystemExit(f"Expected float32 input, got {X.dtype}")
    if not np.isfinite(X).all():
        raise SystemExit("Non-finite values in GRF input tensor")

    feature_cols = get_feature_columns()
    C_canon = len(feature_cols)
    C_actual = int(X.shape[2])

    if X.shape != (int(NUM_WINDOWS), int(WINDOW_LEN), C_canon):
        raise SystemExit(
            f"Unexpected adapter shape; got {tuple(X.shape)}, "
            f"expected ({int(NUM_WINDOWS)}, {int(WINDOW_LEN)}, {C_canon})"
        )
    if C_actual != C_canon:
        raise SystemExit(f"C_actual={C_actual} does not match C_canon={C_canon}")

    sensor_slices = get_sensor_slices(feature_cols)

    model_obj: Any = None
    if have_vnext:
        try:
            model_obj = VNextFzModel(  # type: ignore[operator]
                in_channels=C_canon,
                sensor_slices=sensor_slices,
            )
        except Exception:  # pragma: no cover - defensive
            model_obj = None

    if model_obj is None:
        # Fallback tiny CPU-only model mapping (B, T, C) → (B, T, 1), deterministic
        # under the seeding above.
        class _TinyGRFModel(torch.nn.Module):  # type: ignore[misc]
            def __init__(self, in_channels: int) -> None:
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(in_channels, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 1),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
                # x: (B, T, C)
                b, t, c = x.shape
                x_flat = x.reshape(b * t, c)
                y_flat = self.net(x_flat)
                y = y_flat.reshape(b, t, 1)
                return y

        model_obj = _TinyGRFModel(in_channels=C_canon)

    model_obj = model_obj.cpu().eval()

    with torch.no_grad():
        y = model_obj(torch.from_numpy(X).cpu())

    y_np = y.detach().cpu().numpy()

    flat = y_np.reshape(-1).astype(np.float64)
    total = int(flat.size)
    if total == 0:
        finite_fraction = 1.0
    else:
        finite_mask = np.isfinite(flat)
        finite_count = int(finite_mask.sum())
        finite_fraction = float(finite_count) / float(total)

    if finite_fraction != 1.0:
        raise NonFiniteError(finite_fraction)

    return y_np


def _compute_contract() -> Dict[str, Any]:
    y = _run_inference()

    if y.ndim != 3:
        # Treat unexpected rank as a shape/dtype mismatch during comparison.
        shape = [int(d) for d in y.shape]
        raise SystemExit(
            f"Unexpected output rank; got shape={shape}, expected rank-3 (B, T, C)"
        )

    shape = [int(d) for d in y.shape]
    dtype = str(np.dtype(y.dtype))

    flat32 = y.reshape(-1).astype(np.float32)
    total = int(flat32.size)
    if total == 0:
        finite_fraction = 1.0
    else:
        finite_mask = np.isfinite(flat32)
        finite_count = int(finite_mask.sum())
        finite_fraction = float(finite_count) / float(total)

    if finite_fraction != 1.0:
        raise NonFiniteError(finite_fraction)

    flat64 = flat32.astype(np.float64)

    # Global aggregate stats over the entire output tensor.
    global_mean = float(flat64.mean()) if flat64.size else 0.0
    global_std = float(flat64.std()) if flat64.size else 0.0

    # Stable fingerprint metrics for quick visual comparison.
    if flat64.size:
        mean_abs = float(np.mean(np.abs(flat64)))
        p50 = float(np.percentile(flat64, 50.0))
        p95 = float(np.percentile(flat64, 95.0))
    else:
        mean_abs = 0.0
        p50 = 0.0
        p95 = 0.0

    if flat32.size:
        fp_payload = ",".join(f"{float(v):.6f}" for v in flat32)
        output_fingerprint = hashlib.sha256(fp_payload.encode("utf-8")).hexdigest()
    else:
        output_fingerprint = hashlib.sha256(b"").hexdigest()

    b, t, c = y.shape
    per_channel_stats: Dict[str, Dict[str, float]] = {}
    y64 = y.astype(np.float64)
    for ch in range(c):
        vals = y64[:, :, ch].reshape(-1)
        per_channel_stats[str(ch)] = {
            "min": float(vals.min()),
            "max": float(vals.max()),
            "mean": float(vals.mean()),
            "std": float(vals.std()),
        }

    contract: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "fixture": FIXTURE_REL_PATH,
        "shape": shape,
        "dtype": dtype,
        "output_shape": shape,
        "output_dtype": dtype,
        "finite_fraction": float(finite_fraction),
        "global_mean": float(global_mean),
        "global_std": float(global_std),
        "mean_abs": float(mean_abs),
        "p50": float(p50),
        "p95": float(p95),
        "output_fingerprint": output_fingerprint,
        "fingerprint": {
            "mean_abs": float(mean_abs),
            "p50": float(p50),
            "p95": float(p95),
        },
        "per_channel_stats": per_channel_stats,
        "output_flat": [float(v) for v in flat64],
    }

    return contract


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"IMU_TO_GRF_CONTRACT missing baseline JSON: {path}")
        raise SystemExit(2)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        print(f"IMU_TO_GRF_CONTRACT invalid baseline JSON {path}: {exc}")
        raise SystemExit(2)
    if not isinstance(data, dict):
        print(f"IMU_TO_GRF_CONTRACT baseline JSON is not an object: {path}")
        raise SystemExit(2)
    return data


def _handle_nonfinite(exc: NonFiniteError, regen_cmd: str, print_regen_cmd: bool) -> int:
    finite_fraction = float(exc.finite_fraction)
    print("FAIL imu_to_grf_contract: non-finite values")
    print(f"finite_fraction={finite_fraction:.9f}")
    # Always print regen hint on failure.
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
    print(f"IMU_TO_GRF_CONTRACT baseline: {baseline_path}")
    print(f"IMU_TO_GRF_CONTRACT fixture: {FIXTURE_REL_PATH}")

    base_shape = list(map(int, baseline.get("shape", [])))
    curr_shape = list(map(int, current.get("shape", [])))
    base_dtype = str(baseline.get("dtype"))
    curr_dtype = str(current.get("dtype"))

    if base_shape != curr_shape or base_dtype != curr_dtype:
        print("FAIL imu_to_grf_contract: shape/dtype mismatch")
        print(f"baseline_shape={base_shape}")
        print(f"current_shape={curr_shape}")
        print(f"baseline_dtype={base_dtype}")
        print(f"current_dtype={curr_dtype}")
        print(f"REGEN_BASELINE_CMD={regen_cmd}")
        return EXIT_SHAPE_DTYPE_MISMATCH

    base_finite = float(baseline.get("finite_fraction", 0.0))
    curr_finite = float(current.get("finite_fraction", 0.0))
    if base_finite != 1.0 or curr_finite != 1.0:
        print("FAIL imu_to_grf_contract: non-finite values")
        print(f"baseline_finite_fraction={base_finite:.9f}")
        print(f"current_finite_fraction={curr_finite:.9f}")
        print(f"REGEN_BASELINE_CMD={regen_cmd}")
        return EXIT_NONFINITE
    # Global stats and fingerprint comparisons.
    base_mean = float(baseline.get("global_mean", 0.0))
    curr_mean = float(current.get("global_mean", 0.0))

    base_std = float(baseline.get("global_std", 0.0))
    curr_std = float(current.get("global_std", 0.0))

    base_mean_abs = float(baseline.get("mean_abs", 0.0))
    curr_mean_abs = float(current.get("mean_abs", 0.0))

    base_p50 = float(baseline.get("p50", 0.0))
    curr_p50 = float(current.get("p50", 0.0))

    base_p95 = float(baseline.get("p95", 0.0))
    curr_p95 = float(current.get("p95", 0.0))

    base_fp_str = str(baseline.get("output_fingerprint", ""))
    curr_fp_str = str(current.get("output_fingerprint", ""))

    base_flat = np.asarray(baseline.get("output_flat", []), dtype=np.float64)
    curr_flat = np.asarray(current.get("output_flat", []), dtype=np.float64)

    if base_flat.shape != curr_flat.shape:
        print("FAIL imu_to_grf_contract: shape/dtype mismatch")
        print(f"baseline_output_flat_shape={base_flat.shape}")
        print(f"current_output_flat_shape={curr_flat.shape}")
        print(f"REGEN_BASELINE_CMD={regen_cmd}")
        return EXIT_SHAPE_DTYPE_MISMATCH

    if base_flat.size == 0:
        rmse = 0.0
    else:
        diff = curr_flat - base_flat
        rmse = float(np.sqrt(np.mean(diff * diff)))

    # Summary line for current run with key metrics and RMSE vs baseline.
    print(
        "IMU_TO_GRF_CONTRACT current: "
        f"shape={curr_shape} "
        f"dtype={curr_dtype} "
        f"finite_fraction={curr_finite:.9f} "
        f"global_mean={curr_mean:.6g} "
        f"global_std={curr_std:.6g} "
        f"mean_abs={curr_mean_abs:.6g} "
        f"p50={curr_p50:.6g} "
        f"p95={curr_p95:.6g} "
        f"global_rmse={rmse:.6g}"
    )

    # Anti-degenerate invariants on the current output.
    invariant_violation = False
    if curr_std < 1e-6:
        invariant_violation = True
    if curr_p95 < curr_p50:
        invariant_violation = True
    if curr_std < 1e-4:
        if abs(curr_p95 - curr_p50) >= 1e-6 or abs(curr_mean_abs) >= 1e-3:
            invariant_violation = True

    if invariant_violation:
        print("FAIL imu_to_grf_contract: invariant violation")
        print(
            "INVARIANTS current: "
            f"finite_fraction={curr_finite:.9f} "
            f"global_std={curr_std:.6g} "
            f"mean_abs={curr_mean_abs:.6g} "
            f"p50={curr_p50:.6g} "
            f"p95={curr_p95:.6g}"
        )
        print(f"REGEN_BASELINE_CMD={regen_cmd}")
        return EXIT_CONTRACT_MISMATCH

    # Fingerprint equality check.
    if base_fp_str != curr_fp_str:
        print("FAIL imu_to_grf_contract: fingerprint mismatch")
        print(f"FINGERPRINT baseline={base_fp_str}")
        print(f"FINGERPRINT current={curr_fp_str}")
        print(f"REGEN_BASELINE_CMD={regen_cmd}")
        return EXIT_CONTRACT_MISMATCH

    # Stats drift checks with tight absolute tolerances.
    drift_records = []

    def _check_stat(name: str, base_val: float, curr_val: float) -> None:
        abs_diff = abs(curr_val - base_val)
        if abs_diff > ABS_TOL:
            drift_records.append((name, base_val, curr_val, abs_diff))

    _check_stat("global_mean", base_mean, curr_mean)
    _check_stat("global_std", base_std, curr_std)
    _check_stat("mean_abs", base_mean_abs, curr_mean_abs)
    _check_stat("p50", base_p50, curr_p50)
    _check_stat("p95", base_p95, curr_p95)

    if drift_records:
        print("FAIL imu_to_grf_contract: stats drift")
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

    print("PASS imu_to_grf_contract")
    if print_regen_cmd:
        print(f"REGEN_BASELINE_CMD={regen_cmd}")
    return 0


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute or check the IMU→GRF inference non-regression contract "
            "on the canonical IMU sample fixture."
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
            "In check mode, print a helper command for regenerating the "
            "baseline JSON (does not modify files)."
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

    # mode == "check"
    baseline = _load_json(baseline_path)

    try:
        current = _compute_contract()
    except NonFiniteError as exc:
        code = _handle_nonfinite(
            exc,
            regen_cmd,
            print_regen_cmd=getattr(args, "print_regen_cmd", False),
        )
        raise SystemExit(code)

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
