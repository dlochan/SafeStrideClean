#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


JSON_PATH = Path(
    "data/vnext_gt_real_out/vnext_fz/20260119-033819_df2881d2/analysis_eval/3d_metrics_summary.json"
)


def main() -> int:
    if not JSON_PATH.is_file():
        print(f"FAIL overfit3d_contract: missing summary JSON at {JSON_PATH}")
        return 2

    try:
        with JSON_PATH.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception as e:
        print(f"FAIL overfit3d_contract: failed to load JSON: {e}")
        return 2

    reasons = []

    units_detected = (payload.get("units_detected") or "").strip().lower()
    num_windows = payload.get("num_windows")
    window_len = payload.get("window_len")
    axis_summaries = payload.get("axis_summaries") or {}
    gate = payload.get("gate") or {}

    fz = axis_summaries.get("Fz") or axis_summaries.get("fz") or {}
    fz_rmse = fz.get("rmse")
    fz_corr = fz.get("corr")

    if units_detected not in {"newtons", "newton", "n"}:
        reasons.append(f"units_detected={units_detected!r} not in {{'newtons','newton','n'}}")

    if num_windows != 64:
        reasons.append(f"num_windows={num_windows!r} != 64")

    if window_len != 256:
        reasons.append(f"window_len={window_len!r} != 256")

    if fz_rmse is None:
        reasons.append("axis_summaries.Fz.rmse missing")
    elif float(fz_rmse) > 150.0:
        reasons.append(f"axis_summaries.Fz.rmse={float(fz_rmse):.6f} > 150.0")

    if fz_corr is None:
        reasons.append("axis_summaries.Fz.corr missing")
    elif float(fz_corr) < 0.90:
        reasons.append(f"axis_summaries.Fz.corr={float(fz_corr):.6f} < 0.90")

    gate_status = gate.get("status")
    if gate_status != "PASS":
        reasons.append(f"gate.status={gate_status!r} != 'PASS'")

    if reasons:
        print("FAIL overfit3d_contract:")
        for r in reasons:
            print(f"  - {r}")
        return 2

    print(
        "PASS overfit3d_contract: "
        f"units={units_detected} "
        f"windows={num_windows} "
        f"len={window_len} "
        f"Fz_rmse={float(fz_rmse):.6f} "
        f"Fz_corr={float(fz_corr):.6f}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
