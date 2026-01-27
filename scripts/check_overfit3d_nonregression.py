#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def pick(d, paths):
    for path in paths:
        cur = d
        ok = True
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if ok:
            return cur
    return None

def main():
    if len(sys.argv) != 3:
        print("Usage: check_overfit3d_nonregression.py <ANALYZER_JSON> <BASELINE_JSON>")
        return 2

    analyzer_path = Path(sys.argv[1])
    baseline_path = Path(sys.argv[2])

    baseline_raw = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline = baseline_raw.get("extract", {})
    need = ["units", "windows", "len", "Fz_rmse", "Fz_corr"]
    missing_base = [k for k in need if k not in baseline]
    if missing_base:
        print(f"FAIL non-regression: baseline missing keys {missing_base}")
        return 20

    cur_raw = json.loads(analyzer_path.read_text(encoding="utf-8"))

    # Current repo stores Fz metrics at axis_summaries/Fz with rmse/corr
    cur = {
        "units": "newtons" if str(cur_raw.get("units_detected","newtons")).lower().startswith("newton") else str(cur_raw.get("units_detected","")),
        "windows": cur_raw.get("num_windows"),
        "len": cur_raw.get("window_len"),
        "Fz_rmse": pick(cur_raw, [["axis_summaries","Fz","rmse"]]),
        "Fz_corr": pick(cur_raw, [["axis_summaries","Fz","corr"]]),
    }

    missing_cur = [k for k,v in cur.items() if v is None]
    if missing_cur:
        print(f"FAIL non-regression: missing current keys {missing_cur}")
        return 21

    # Tolerances
    rmse_tol_frac = 0.05   # RMSE can increase up to +5%
    corr_tol_drop = 0.002  # Corr can drop up to -0.002

    base_rmse = float(baseline["Fz_rmse"])
    base_corr = float(baseline["Fz_corr"])
    cur_rmse = float(cur["Fz_rmse"])
    cur_corr = float(cur["Fz_corr"])

    # Hard compatibility checks
    if str(cur["units"]) != str(baseline["units"]):
        print(f"FAIL non-regression: units mismatch baseline={baseline['units']} current={cur['units']}")
        return 24
    if int(cur["len"]) != int(baseline["len"]):
        print(f"FAIL non-regression: window_len mismatch baseline={baseline['len']} current={cur['len']}")
        return 25
    if int(cur["windows"]) != int(baseline["windows"]):
        print(f"FAIL non-regression: num_windows mismatch baseline={baseline['windows']} current={cur['windows']}")
        return 26

    rmse_ok = cur_rmse <= base_rmse * (1.0 + rmse_tol_frac)
    corr_ok = cur_corr >= base_corr - corr_tol_drop

    print(f"NON_REGRESSION baseline: rmse={base_rmse:.6f} corr={base_corr:.6f} windows={baseline['windows']} len={baseline['len']} units={baseline['units']}")
    print(f"NON_REGRESSION current:   rmse={cur_rmse:.6f} corr={cur_corr:.6f} windows={cur['windows']} len={cur['len']} units={cur['units']}")
    print(f"NON_REGRESSION tolerances: rmse<=+{rmse_tol_frac*100:.1f}% corr>=-{corr_tol_drop:.3f}")

    if not rmse_ok:
        print("FAIL non-regression: RMSE regressed beyond tolerance")
        return 22
    if not corr_ok:
        print("FAIL non-regression: Corr regressed beyond tolerance")
        return 23

    print("PASS non-regression")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
