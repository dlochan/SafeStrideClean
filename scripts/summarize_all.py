# scripts/summarize_all.py
import argparse
import json
import os
import glob
from pathlib import Path
from typing import Tuple, List

import pandas as pd

try:
    from src.config import OUT_ROOT as DEFAULT_OUT_ROOT
except Exception:
    DEFAULT_OUT_ROOT = Path("out_grid")

# Reuse the parsing from summarize_subject to ensure consistency
try:
    from scripts.summarize_subject import parse_combo_name
except Exception:
    def parse_combo_name(combo: str) -> Tuple[str, str, str, str, int]:
        parts = combo.split("_")
        if len(parts) < 6:
            raise ValueError(f"Unexpected folder name format: {combo}")
        subject = parts[0]
        trial = "_".join(parts[0:3])
        sensor_set = parts[-3]
        model = parts[-2]
        window = parts[-1]
        if not window.startswith("w"):
            raise ValueError(f"Unexpected window token in '{combo}': {window}")
        wn = window[1:]
        if wn.endswith("ms"):
            wn = wn[:-2]
        window_ms = int(wn)
        return subject, trial, sensor_set.lower(), model.lower(), window_ms


essential_cols = [
    "subject", "trial", "sensor_set", "model", "window_ms",
    "rmse_%BW", "mae_%BW", "metrics_json"
]


def _scan_all(out_root: Path) -> List[dict]:
    rows: List[dict] = []
    pattern = os.path.join(str(out_root), "**", "eval", "metrics_eval.json")
    for metrics_path in glob.glob(pattern, recursive=True):
        combo_dir = Path(metrics_path).parent.parent.name
        try:
            subj, trial, sensor_set, model, window_ms = parse_combo_name(combo_dir)
        except Exception as e:
            # Skip unrecognized folders
            print(f"[SKIP] {combo_dir}: {e}")
            continue
        try:
            with open(metrics_path, "r") as f:
                m = json.load(f)
        except Exception as e:
            print(f"[SKIP] cannot read {metrics_path}: {e}")
            continue
        rows.append({
            "subject": subj,
            "trial": trial,
            "sensor_set": sensor_set,
            "model": model,
            "window_ms": window_ms,
            "rmse_%BW": m.get("rmse_%BW"),
            "mae_%BW": m.get("mae_%BW"),
            "metrics_json": metrics_path,
        })
    return rows


def build_global_leaderboard(out_root: Path, save_dir: Path | None = None):
    out_root = Path(out_root)
    save_dir = Path.cwd() if save_dir is None else Path(save_dir)
    rows = _scan_all(out_root)
    df = pd.DataFrame(rows)

    leaderboard_csv = save_dir / "out_grid_leaderboard.csv"
    dashboard_json = save_dir / "out_grid_dashboard.json"

    if df.empty:
        df.to_csv(leaderboard_csv, index=False)
        with open(dashboard_json, "w") as f:
            json.dump({"subjects": {}, "overall": {}}, f, indent=2)
        return df, leaderboard_csv, dashboard_json, {"subjects": {}, "overall": {}}

    # keep only rows with rmse
    df = df[df["rmse_%BW"].notna()].copy()
    df = df.sort_values(["subject", "trial", "rmse_%BW"], ascending=[True, True, True])

    # write leaderboard
    df.to_csv(leaderboard_csv, index=False)

    # dashboard summary
    dash = {"subjects": {}, "overall": {}}
    for subj, g in df.groupby("subject"):
        dash["subjects"][subj] = {
            "n_rows": int(len(g)),
            "n_trials": int(g["trial"].nunique()),
            "best_rmse_%BW": float(g["rmse_%BW"].min()),
            "mean_rmse_%BW": float(g["rmse_%BW"].mean()),
        }
    dash["overall"] = {
        "n_rows": int(len(df)),
        "n_subjects": int(df["subject"].nunique()),
        "n_trials": int(df["trial"].nunique()),
        "mean_rmse_%BW": float(df["rmse_%BW"].mean()),
        "median_rmse_%BW": float(df["rmse_%BW"].median()),
    }

    with open(dashboard_json, "w") as f:
        json.dump(dash, f, indent=2)

    return df, leaderboard_csv, dashboard_json, dash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT), help="Root of grid outputs")
    ap.add_argument("--save_dir", default=".", help="Directory to write leaderboard and dashboard")
    args = ap.parse_args()

    df, lb_path, dash_path, _ = build_global_leaderboard(Path(args.out_root), Path(args.save_dir))
    print(f"[OK] global leaderboard → {lb_path} (rows={len(df)})")
    print(f"[OK] dashboard → {dash_path}")


if __name__ == "__main__":
    main()
