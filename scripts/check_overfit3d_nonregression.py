#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def pick(d, path):
    cur = d
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur

def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def main():
    if len(sys.argv) != 3:
        print("usage: check_overfit3d_nonregression.py <ANALYZER_JSON> <BASELINE_JSON>", file=sys.stderr)
        return 2

    analyzer_path = Path(sys.argv[1])
    baseline_path = Path(sys.argv[2])

    base_raw = load_json(baseline_path)
    base = base_raw.get("extract", {})
    need_base = ["units", "windows", "len", "Fz_rmse", "Fz_corr"]
    missing_base = [k for k in need_base if k not in base]
    if missing_base:
        print(f"FAIL non-regression: baseline missing keys {missing_base}", file=sys.stderr)
        return 20

    cur_raw = load_json(analyzer_path)

    # Current metrics live in axis_summaries/Fz (your confirmed structure)
    cur_rmse = pick(cur_raw, ["axis_summaries", "Fz", "rmse"])
    cur_corr = pick(cur_raw, ["axis_summaries", "Fz", "corr"])
    cur_windows = cur_raw.get("num_windows")
    cur_len = cur_raw.get("window_len")
    units_detected = cur_raw.get("units_detected", "")
    cur_units = "newtons" if "newton" in str(units_detected).lower() else str(units_detected)

    missing_cur = []
    if cur_rmse is None: missing_cur.append("axis_summaries/Fz/rmse")
    if cur_corr is None: missing_cur.append("axis_summaries/Fz/corr")
    if cur_windows is None: missing_cur.append("num_windows")
    if cur_len is None: missing_cur.append("window_len")
    if missing_cur:
        print(f"FAIL non-regression: missing current keys {missing_cur}", file=sys.stderr)
        return 21

    # Non-regression tolerances
    rmse_tol_frac = 0.05
    corr_tol_drop = 0.002

    base_rmse = float(base["Fz_rmse"])
    base_corr = float(base["Fz_corr"])
    cur_rmse = float(cur_rmse)
    cur_corr = float(cur_corr)

    rmse_ok = cur_rmse <= base_rmse * (1.0 + rmse_tol_frac)
    corr_ok = cur_corr >= base_corr - corr_tol_drop

    # Hard sanity: ensure the shape matches (windows/len). Units mismatch is a hard fail.
    if int(cur_windows) != int(base["windows"]) or int(cur_len) != int(base["len"]):
        print(f"FAIL non-regression: windows/len mismatch baseline(windows={base['windows']} len={base['len']}) "
              f"current(windows={cur_windows} len={cur_len})", file=sys.stderr)
        return 24
    if str(cur_units) != str(base["units"]):
        print(f"FAIL non-regression: units mismatch baseline(units={base['units']}) current(units={cur_units})", file=sys.stderr)
        return 25

    print(f"NON_REGRESSION baseline: rmse={base_rmse:.6f} corr={base_corr:.6f} windows={base['windows']} len={base['len']} units={base['units']}")
    print(f"NON_REGRESSION current:   rmse={cur_rmse:.6f} corr={cur_corr:.6f} windows={cur_windows} len={cur_len} units={cur_units}")
    print(f"NON_REGRESSION tolerances: rmse<=+{rmse_tol_frac*100:.1f}% corr>=-{corr_tol_drop:.3f}")

    if not rmse_ok:
        print("FAIL non-regression: RMSE regressed beyond tolerance", file=sys.stderr)
        return 22
    if not corr_ok:
        print("FAIL non-regression: Corr regressed beyond tolerance", file=sys.stderr)
        return 23

    print("PASS non-regression")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
