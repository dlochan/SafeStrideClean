#!/usr/bin/env bash
set -euo pipefail

# Smoke entrypoint for the MP_CONVERGE_3D work.
# - Verifies the 64-window overfit contract.
# - Prints a one-screen summary for:
#     * 64-window subset eval
#     * full-manifest generalization eval
#
# This script is read-only: it only inspects existing JSON artifacts.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"
cd "$ROOT"

SUBSET_JSON="data/vnext_gt_real_out/vnext_fz/20260119-033819_df2881d2/analysis_eval/3d_metrics_summary.json"
FULL_JSON="data/vnext_gt_real_out/vnext_fz/20260119-033819_df2881d2/analysis_eval_full/3d_metrics_summary.json"

echo "[mp_converge_3d_smoke] Running overfit3d contract check..." >&2
python3 scripts/check_overfit3d_contract.py

echo "[mp_converge_3d_smoke] Summarizing subset and full-manifest metrics..." >&2

python3 - << 'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

SUBSET_JSON = Path("data/vnext_gt_real_out/vnext_fz/20260119-033819_df2881d2/analysis_eval/3d_metrics_summary.json")
FULL_JSON = Path("data/vnext_gt_real_out/vnext_fz/20260119-033819_df2881d2/analysis_eval_full/3d_metrics_summary.json")

missing = []
for p in (SUBSET_JSON, FULL_JSON):
    if not p.is_file():
        missing.append(str(p))

if missing:
    print("SMOKE ERROR: missing required JSON artifact(s):", file=sys.stderr)
    for m in missing:
        print(f"  - {m}", file=sys.stderr)
    raise SystemExit(1)


def load_summary(path: Path):
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    units = (payload.get("units_detected") or "").strip().lower()
    num_windows = payload.get("num_windows")
    window_len = payload.get("window_len")
    axis_summaries = payload.get("axis_summaries") or {}
    gate = payload.get("gate") or {}
    fz = axis_summaries.get("Fz") or axis_summaries.get("fz") or {}
    fz_rmse = fz.get("rmse")
    fz_corr = fz.get("corr")

    # Contract-style PASS/FAIL: require Newton units and strong Fz.
    ok = True
    reasons = []
    if units not in {"newtons", "newton", "n"}:
        ok = False
        reasons.append(f"units={units!r} not newton(s)")
    if fz_rmse is None or float(fz_rmse) > 150.0:
        ok = False
        reasons.append(f"Fz_rmse={fz_rmse!r} > 150.0")
    if fz_corr is None or float(fz_corr) < 0.90:
        ok = False
        reasons.append(f"Fz_corr={fz_corr!r} < 0.90")

    return {
        "units": units,
        "num_windows": num_windows,
        "window_len": window_len,
        "fz_rmse": float(fz_rmse) if fz_rmse is not None else float("nan"),
        "fz_corr": float(fz_corr) if fz_corr is not None else float("nan"),
        "gate_status": gate.get("status"),
        "pass_contract": ok,
        "reasons": reasons,
    }

subset = load_summary(SUBSET_JSON)
full = load_summary(FULL_JSON)

print("SMOKE 64-window: {status} units={units} windows={w} len={L} Fz_rmse={rmse:.6f} Fz_corr={corr:.6f}".format(
    status="PASS" if subset["pass_contract"] else "FAIL",
    units=subset["units"],
    w=subset["num_windows"],
    L=subset["window_len"],
    rmse=subset["fz_rmse"],
    corr=subset["fz_corr"],
))

print("SMOKE full-manifest: {status} units={units} windows={w} len={L} Fz_rmse={rmse:.6f} Fz_corr={corr:.6f}".format(
    status="PASS" if full["pass_contract"] else "FAIL",
    units=full["units"],
    w=full["num_windows"],
    L=full["window_len"],
    rmse=full["fz_rmse"],
    corr=full["fz_corr"],
))

PY
