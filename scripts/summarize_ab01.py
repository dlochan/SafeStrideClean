# scripts/summarize_ab01.py
import argparse, json, os, glob, sys
import pandas as pd

def parse_combo(combo: str):
    """
    Expected dir pattern examples:
      AB01_cutting_leftfast_lpelvis_hgb_w300
      AB01_cutting_leftslow_rshank_rf_w200
    Returns: subject, trial, sensor, model, window_ms
    """
    parts = combo.split("_")
    # We expect ..._<sensor>_<model>_<wXXX>
    if len(parts) < 6:
        raise ValueError(f"Unexpected folder name format: {combo}")

    subject = parts[0]                  # e.g., AB01
    trial   = "_".join(parts[:3])       # e.g., AB01_cutting_leftfast
    sensor  = parts[-3]                 # e.g., lshank / rshank / lpthigh / rpthigh / lpelvis / rpelvis
    model   = parts[-2]                 # e.g., rf / hgb / ridge
    window  = parts[-1]                 # e.g., w300 or w300ms

    if not window.startswith("w"):
        raise ValueError(f"Unexpected window token in '{combo}': {window}")
    window_num = window[1:]
    if window_num.endswith("ms"):
        window_num = window_num[:-2]
    window_ms = int(window_num)

    return subject, trial, sensor, model, window_ms

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="out_grid",
                    help="Root folder containing per-run directories with eval/metrics_eval.json")
    ap.add_argument("--save_csv", default="out_grid_leaderboard_AB01.csv",
                    help="Where to write the leaderboard CSV")
    ap.add_argument("--valid_sensors", default="lpthigh,rpthigh,lshank,rshank",
                    help="Comma-separated whitelist of sensors to keep (default: knee-only)")
    ap.add_argument("--strict", action="store_true",
                    help="If set, raise on unexpected folder names instead of skipping")
    args = ap.parse_args()

    valid_sensors = {s.strip().lower() for s in args.valid_sensors.split(",") if s.strip()}

    rows = []
    pattern = os.path.join(args.out_root, "**", "eval", "metrics_eval.json")
    hits = glob.glob(pattern, recursive=True)

    if not hits:
        print(f"[WARN] No metrics files found under: {pattern}")
    for metrics_path in hits:
        # example path: .../out_grid/AB01_cutting_leftfast_lpelvis_hgb_w300/eval/metrics_eval.json
        combo_dir = os.path.normpath(metrics_path).split(os.sep)[-3]
        try:
            subj, trial, sensor, model, window_ms = parse_combo(combo_dir)
        except Exception as e:
            msg = f"[SKIP] {combo_dir}: {e}"
            if args.strict:
                raise
            else:
                print(msg)
                continue

        # Read metrics
        try:
            with open(metrics_path, "r") as f:
                m = json.load(f)
        except Exception as e:
            msg = f"[SKIP] Cannot read metrics: {metrics_path} ({e})"
            if args.strict:
                raise
            else:
                print(msg)
                continue

        rows.append({
            "subject": subj,
            "trial": trial,
            "sensor": sensor.lower(),
            "model": model.lower(),
            "window_ms": window_ms,
            "rmse_%BW": m.get("rmse_%BW"),
            "mae_%BW": m.get("mae_%BW"),
            "metrics_json": metrics_path
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("[WARN] No rows collected; nothing to write.")
        sys.exit(0)

    # Keep only the sensors we actually can deploy (knee-only by default)
    if valid_sensors:
        before = len(df)
        df = df[df["sensor"].isin(valid_sensors)].copy()
        after = len(df)
        print(f"[INFO] Filtered by sensors {sorted(valid_sensors)}: kept {after}/{before} rows")

    # Sort nicely
    df = df.sort_values(["trial", "rmse_%BW"], ascending=[True, True])

    # Write leaderboard
    df.to_csv(args.save_csv, index=False)
    print(f"[OK] wrote {args.save_csv} with {len(df)} rows")

if __name__ == "__main__":
    main()
