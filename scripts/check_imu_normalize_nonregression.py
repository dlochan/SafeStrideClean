#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


SCHEMA_VERSION = "imu_normalize_contract_v1"
FIXTURE_REL_PATH = "tests/fixtures/imu_messy.csv"
ABS_TOL = 1e-6

EXIT_CONTRACT_MISMATCH = 41
EXIT_NONFINITE = 42
EXIT_SCHEMA_MISMATCH = 43


class SchemaMismatchError(Exception):
    def __init__(self, expected: List[str], actual: List[str]) -> None:
        super().__init__("schema mismatch")
        self.expected = expected
        self.actual = actual


class DTypeMismatchError(Exception):
    def __init__(self, bad_dtypes: Dict[str, str]) -> None:
        super().__init__("dtype mismatch")
        self.bad_dtypes = bad_dtypes


class NonFiniteError(Exception):
    def __init__(self, finite_fraction: float) -> None:
        super().__init__("non-finite values")
        self.finite_fraction = finite_fraction


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_sys_path() -> None:
    root = _repo_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def _load_normalized_df() -> Tuple[pd.DataFrame, List[str]]:
    """Normalize the messy fixture and enforce schema/dtype/finite invariants.

    Returns
    -------
    (df, feature_cols)
        df is the normalized canonical DataFrame.
        feature_cols is the ordered list of canonical feature columns.
    """

    _ensure_sys_path()

    from src.adapters.imu_normalize import (  # type: ignore
        normalize_imu_csv_to_canon_df,
    )
    from src.vnext.data.imu_schema import get_feature_columns  # type: ignore

    fixture_path = _repo_root() / FIXTURE_REL_PATH
    df = normalize_imu_csv_to_canon_df(str(fixture_path))
    feature_cols = list(get_feature_columns())

    # Enforce exact canonical columns (order + names).
    cols = list(df.columns)
    if cols != feature_cols:
        raise SchemaMismatchError(feature_cols, cols)

    # Enforce float32 dtype for every canonical column.
    bad_dtypes: Dict[str, str] = {}
    for name in feature_cols:
        dt = np.dtype(df[name].dtype)
        if dt != np.float32:
            bad_dtypes[name] = str(dt)
    if bad_dtypes:
        raise DTypeMismatchError(bad_dtypes)

    # Enforce finiteness of all values, computing fraction in float64.
    values64 = df.to_numpy(dtype=np.float64, copy=False)
    total = int(values64.size)
    if total == 0:
        finite_fraction = 1.0
    else:
        finite_mask = np.isfinite(values64)
        finite_count = int(finite_mask.sum())
        finite_fraction = float(finite_count / total)

    if finite_fraction != 1.0:
        raise NonFiniteError(finite_fraction)

    return df, feature_cols


def _compute_stats(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    values64 = df[feature_cols].to_numpy(dtype=np.float64, copy=False)
    rows, cols = values64.shape

    per_col_stats: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(feature_cols):
        col_vals = values64[:, idx]
        col_min = float(np.min(col_vals))
        col_max = float(np.max(col_vals))
        col_mean = float(np.mean(col_vals))
        col_std = float(np.std(col_vals))  # population std (ddof=0)
        per_col_stats[name] = {
            "min": col_min,
            "max": col_max,
            "mean": col_mean,
            "std": col_std,
        }

    flat = values64.reshape(-1)
    global_min = float(np.min(flat))
    global_max = float(np.max(flat))
    global_mean = float(np.mean(flat))
    global_std = float(np.std(flat))

    # All values already validated finite in _load_normalized_df.
    finite_fraction = 1.0

    dtypes: Dict[str, str] = {}
    for name in feature_cols:
        dtypes[name] = str(np.dtype(df[name].dtype))

    contract: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "fixture": FIXTURE_REL_PATH,
        "canon_cols": list(feature_cols),
        "shape": [int(rows), int(cols)],
        "dtypes": dtypes,
        "finite_fraction": float(finite_fraction),
        "per_col_stats": per_col_stats,
        "global_stats": {
            "min": global_min,
            "max": global_max,
            "mean": global_mean,
            "std": global_std,
        },
    }
    return contract


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"IMU_NORMALIZE_CONTRACT missing baseline JSON: {path}")
        raise SystemExit(2)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        print(f"IMU_NORMALIZE_CONTRACT invalid baseline JSON {path}: {exc}")
        raise SystemExit(2)
    if not isinstance(data, dict):
        print(f"IMU_NORMALIZE_CONTRACT baseline JSON is not an object: {path}")
        raise SystemExit(2)
    return data


def _compare_contracts(
    baseline: Dict[str, Any], current: Dict[str, Any]
) -> Tuple[bool, Dict[str, Any]]:
    ok = True

    if baseline.get("schema_version") != current.get("schema_version"):
        ok = False

    if baseline.get("fixture") != current.get("fixture"):
        ok = False

    if baseline.get("canon_cols") != current.get("canon_cols"):
        ok = False

    if baseline.get("shape") != current.get("shape"):
        ok = False

    if baseline.get("dtypes") != current.get("dtypes"):
        ok = False

    # finite_fraction is enforced to be 1.0 before we ever build the
    # contract; no need to compare here beyond structural equality.
    if float(baseline.get("finite_fraction", 0.0)) != float(
        current.get("finite_fraction", 0.0)
    ):
        ok = False

    # Per-column stats with absolute tolerances.
    per_col_base = baseline.get("per_col_stats", {})
    per_col_curr = current.get("per_col_stats", {})

    col_diffs: List[Dict[str, Any]] = []
    canon_cols = baseline.get("canon_cols") or current.get("canon_cols") or []
    for name in canon_cols:
        b_stats = per_col_base.get(name)
        c_stats = per_col_curr.get(name)
        if not isinstance(b_stats, dict) or not isinstance(c_stats, dict):
            ok = False
            continue
        min_diff = abs(float(c_stats.get("min", 0.0)) - float(b_stats.get("min", 0.0)))
        max_diff = abs(float(c_stats.get("max", 0.0)) - float(b_stats.get("max", 0.0)))
        mean_diff = abs(float(c_stats.get("mean", 0.0)) - float(b_stats.get("mean", 0.0)))
        std_diff = abs(float(c_stats.get("std", 0.0)) - float(b_stats.get("std", 0.0)))
        diffs = {
            "name": name,
            "min_diff": min_diff,
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "std_diff": std_diff,
        }
        if (
            min_diff > ABS_TOL
            or max_diff > ABS_TOL
            or mean_diff > ABS_TOL
            or std_diff > ABS_TOL
        ):
            ok = False
            col_diffs.append(diffs)

    # Global stats with the same tolerances.
    base_global = baseline.get("global_stats", {})
    curr_global = current.get("global_stats", {})
    g_min_diff = abs(
        float(curr_global.get("min", 0.0)) - float(base_global.get("min", 0.0))
    )
    g_max_diff = abs(
        float(curr_global.get("max", 0.0)) - float(base_global.get("max", 0.0))
    )
    g_mean_diff = abs(
        float(curr_global.get("mean", 0.0)) - float(base_global.get("mean", 0.0))
    )
    g_std_diff = abs(
        float(curr_global.get("std", 0.0)) - float(base_global.get("std", 0.0))
    )
    if (
        g_min_diff > ABS_TOL
        or g_max_diff > ABS_TOL
        or g_mean_diff > ABS_TOL
        or g_std_diff > ABS_TOL
    ):
        ok = False

    global_diffs = {
        "min_diff": g_min_diff,
        "max_diff": g_max_diff,
        "mean_diff": g_mean_diff,
        "std_diff": g_std_diff,
    }

    return ok, {"col_diffs": col_diffs, "global_diffs": global_diffs}


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute or check the IMU normalizer non-regression contract on the "
            "messy fixture."
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

    regen_cmd = (
        "python scripts/check_imu_normalize_nonregression.py "
        "--mode compute --baseline "
        f"{baseline_path}"
    )

    try:
        df, feature_cols = _load_normalized_df()
    except SchemaMismatchError as exc:
        expected_cols = list(exc.expected)
        actual_cols = list(exc.actual)
        missing = [c for c in expected_cols if c not in actual_cols]
        extra = [c for c in actual_cols if c not in expected_cols]
        print("FAIL imu_normalize_contract: schema mismatch")
        print(f"expected_cols_n={len(expected_cols)}")
        print(f"actual_cols_n={len(actual_cols)}")
        print(f"missing_cols={','.join(missing[:10])}")
        print(f"extra_cols={','.join(extra[:10])}")
        if args.mode == "check" and args.print_regen_cmd:
            print(f"REGEN_BASELINE_CMD={regen_cmd}")
        raise SystemExit(EXIT_SCHEMA_MISMATCH)
    except DTypeMismatchError as exc:
        bad_items = [f"{name}={dt}" for name, dt in sorted(exc.bad_dtypes.items())]
        print("FAIL imu_normalize_contract: dtype mismatch")
        print(f"bad_dtypes={','.join(bad_items[:10])}")
        if args.mode == "check" and args.print_regen_cmd:
            print(f"REGEN_BASELINE_CMD={regen_cmd}")
        raise SystemExit(EXIT_SCHEMA_MISMATCH)
    except NonFiniteError as exc:
        finite_fraction = float(exc.finite_fraction)
        print("FAIL imu_normalize_contract: non-finite values")
        print(f"finite_fraction={finite_fraction:.9f}")
        if args.mode == "check" and args.print_regen_cmd:
            print(f"REGEN_BASELINE_CMD={regen_cmd}")
        raise SystemExit(EXIT_NONFINITE)

    current = _compute_stats(df, feature_cols)

    if args.mode == "compute":
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps(current, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return

    # mode == "check"
    baseline = _load_json(baseline_path)

    ok, diff_info = _compare_contracts(baseline, current)

    print(f"IMU_NORMALIZE_CONTRACT baseline: {baseline_path}")
    print(f"IMU_NORMALIZE_CONTRACT fixture: {FIXTURE_REL_PATH}")

    if not ok:
        col_diffs = diff_info.get("col_diffs", []) or []
        global_diffs = diff_info.get("global_diffs", {}) or {}

        print("FAIL imu_normalize_contract: stats drift")

        if col_diffs:
            # Worst offenders per-metric.
            worst_mean = max(col_diffs, key=lambda d: d.get("mean_diff", 0.0))
            worst_std = max(col_diffs, key=lambda d: d.get("std_diff", 0.0))
            worst_min = max(col_diffs, key=lambda d: d.get("min_diff", 0.0))
            worst_max = max(col_diffs, key=lambda d: d.get("max_diff", 0.0))

            print(
                "worst_mean_diff: col="
                f"{worst_mean.get('name', '<unknown>')} diff="
                f"{float(worst_mean.get('mean_diff', 0.0)):.6g}"
            )
            print(
                "worst_std_diff:  col="
                f"{worst_std.get('name', '<unknown>')} diff="
                f"{float(worst_std.get('std_diff', 0.0)):.6g}"
            )
            print(
                "worst_min_diff:  col="
                f"{worst_min.get('name', '<unknown>')} diff="
                f"{float(worst_min.get('min_diff', 0.0)):.6g}"
            )
            print(
                "worst_max_diff:  col="
                f"{worst_max.get('name', '<unknown>')} diff="
                f"{float(worst_max.get('max_diff', 0.0)):.6g}"
            )

            # Top 5 columns by combined score.
            scored = []
            for d in col_diffs:
                score = max(
                    float(d.get("mean_diff", 0.0)),
                    float(d.get("std_diff", 0.0)),
                    float(d.get("min_diff", 0.0)),
                    float(d.get("max_diff", 0.0)),
                )
                scored.append((d.get("name", "<unknown>"), score))
            scored.sort(key=lambda t: t[1], reverse=True)
            top_items = [f"{name}:{score:.6g}" for name, score in scored[:5]]
            print(f"TOP_DRIFT_COLS={','.join(top_items)}")
        else:
            # No per-column diffs but global stats drifted.
            print("worst_mean_diff: col=<none> diff=0")
            print("worst_std_diff:  col=<none> diff=0")
            print("worst_min_diff:  col=<none> diff=0")
            print("worst_max_diff:  col=<none> diff=0")
            print("TOP_DRIFT_COLS=")

        worst_global_mean_diff = float(global_diffs.get("mean_diff", 0.0))
        worst_global_std_diff = float(global_diffs.get("std_diff", 0.0))
        print(f"worst_global_mean_diff={worst_global_mean_diff:.6g}")
        print(f"worst_global_std_diff={worst_global_std_diff:.6g}")

        if args.print_regen_cmd:
            print(f"REGEN_BASELINE_CMD={regen_cmd}")

        raise SystemExit(EXIT_CONTRACT_MISMATCH)

    print("PASS imu_normalize_contract")

    if args.print_regen_cmd:
        print(f"REGEN_BASELINE_CMD={regen_cmd}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main(sys.argv[1:])
