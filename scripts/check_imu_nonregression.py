from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.imu.contract import compute_contract_summary


def _rel_diff(a: float, b: float, eps: float) -> float:
    return abs(a - b) / max(abs(b), eps)


def _check_array(
    *,
    name: str,
    actual: list[float],
    baseline: list[float],
    rel_tol: float,
    abs_eps: float,
    exit_code: int,
) -> None:
    if len(actual) != len(baseline):
        raise SystemExit(f"{exit_code}: {name} length mismatch actual={len(actual)} baseline={len(baseline)}")

    worst = 0.0
    worst_i = -1
    for i, (a, b) in enumerate(zip(actual, baseline)):
        d = _rel_diff(a, b, abs_eps)
        if d > worst:
            worst = d
            worst_i = i
        if d > rel_tol and abs(a - b) > abs_eps:
            raise SystemExit(
                f"{exit_code}: {name} drift too large at idx={i} actual={a} baseline={b} rel_diff={d} tol={rel_tol}"
            )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--fixture", default="tests/fixtures/imu_sample.csv")
    ap.add_argument("--window-len", type=int, default=3)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--include-magnitude", action="store_true")
    args = ap.parse_args()

    baseline_path = Path(args.baseline)
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    actual = compute_contract_summary(
        args.fixture,
        window_len=int(args.window_len),
        stride=int(args.stride),
        include_magnitude=bool(args.include_magnitude),
    )

    # Hard invariants
    if actual.get("feature_tensor_shape") != baseline.get("feature_tensor_shape"):
        raise SystemExit(
            f"20: shape mismatch actual={actual.get('feature_tensor_shape')} baseline={baseline.get('feature_tensor_shape')}"
        )
    if bool(actual.get("has_nan")):
        raise SystemExit("21: has_nan=true")

    # Drift tolerances
    abs_eps = 1e-6
    _check_array(
        name="channel_mean",
        actual=list(actual.get("channel_mean", [])),
        baseline=list(baseline.get("channel_mean", [])),
        rel_tol=0.05,
        abs_eps=abs_eps,
        exit_code=22,
    )
    _check_array(
        name="channel_std",
        actual=list(actual.get("channel_std", [])),
        baseline=list(baseline.get("channel_std", [])),
        rel_tol=0.05,
        abs_eps=abs_eps,
        exit_code=23,
    )
    _check_array(
        name="channel_min",
        actual=list(actual.get("channel_min", [])),
        baseline=list(baseline.get("channel_min", [])),
        rel_tol=0.10,
        abs_eps=abs_eps,
        exit_code=24,
    )
    _check_array(
        name="channel_max",
        actual=list(actual.get("channel_max", [])),
        baseline=list(baseline.get("channel_max", [])),
        rel_tol=0.10,
        abs_eps=abs_eps,
        exit_code=25,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
