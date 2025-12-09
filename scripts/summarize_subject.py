# scripts/summarize_subject.py
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


def parse_combo_name(combo: str) -> Tuple[str, str, str, str, int]:
    """
    Parse a run folder name like:
      AB01_cutting_leftfast_lpthigh_hgb_w300
    Returns: (subject, trial, sensor_set, model, window_ms)
    """
    parts = combo.split("_")
    if len(parts) < 6:
        raise ValueError(f"Unexpected folder name format: {combo}")

    subject = parts[0]                      # e.g., AB01
    trial = "_".join(parts[0:3])           # e.g., AB01_cutting_leftfast
    sensor_set = parts[-3]                  # e.g., lpthigh / lshank / dual / lpelvis
    model = parts[-2]                       # e.g., rf / hgb / ridge
    window = parts[-1]                      # e.g., w300 or w300ms

    if not window.startswith("w"):
        raise ValueError(f"Unexpected window token in '{combo}': {window}")
    wn = window[1:]
    if wn.endswith("ms"):
        wn = wn[:-2]
    window_ms = int(wn)

    return subject, trial, sensor_set.lower(), model.lower(), window_ms


def _scan_metrics(out_root: Path) -> List[dict]:
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
        rmse_bw = m.get("rmse_%BW")
        mae_bw = m.get("mae_%BW")
        rows.append({
            "subject": subj,
            "trial": trial,
            "sensor_set": sensor_set,
            "model": model,
            "window_ms": window_ms,
            "rmse_%BW": rmse_bw,
            "mae_%BW": mae_bw,
            "metrics_json": metrics_path,
        })
    return rows


essential_cols = [
    "subject", "trial", "sensor_set", "model", "window_ms",
    "rmse_%BW", "mae_%BW", "metrics_json"
]


def build_subject_leaderboard(out_root: Path, subject: str, save_dir: Path | None = None):
    out_root = Path(out_root)
    save_dir = Path.cwd() if save_dir is None else Path(save_dir)
    rows = _scan_metrics(out_root)
    df = pd.DataFrame(rows)
    if df.empty:
        # Write empty artifacts for consistency
        lb_path = save_dir / f"out_grid_leaderboard_{subject}.csv"
        df.to_csv(lb_path, index=False)
        sl_path = save_dir / f"shortlist_{subject}.csv"
        df.to_csv(sl_path, index=False)
        return df, pd.DataFrame(), lb_path, sl_path

    # Filter subject and drop rows without rmse_%BW
    df = df[(df["subject"] == subject) & df["rmse_%BW"].notna()].copy()
    if df.empty:
        lb_path = save_dir / f"out_grid_leaderboard_{subject}.csv"
        df.to_csv(lb_path, index=False)
        sl_path = save_dir / f"shortlist_{subject}.csv"
        pd.DataFrame(columns=["trial", "sensor_set", "model", "window_ms"]).to_csv(sl_path, index=False)
        return df, pd.DataFrame(), lb_path, sl_path

    df = df.sort_values(["trial", "rmse_%BW"], ascending=[True, True])

    # Save leaderboard
    lb_path = save_dir / f"out_grid_leaderboard_{subject}.csv"
    df.to_csv(lb_path, index=False)

    # Build shortlist: top-1 per trial, or top-2 if tie on rmse_%BW
    shortlist_rows = []
    for trial, grp in df.groupby("trial", as_index=False):
        min_rmse = grp["rmse_%BW"].min()
        tied = grp[grp["rmse_%BW"] == min_rmse]
        if len(tied) > 2:
            tied = tied.head(2)
        shortlist_rows.append(tied.iloc[0])
        if len(tied) == 2:
            shortlist_rows.append(tied.iloc[1])
    shortlist = pd.DataFrame(shortlist_rows)
    shortlist = shortlist[["trial", "sensor_set", "model", "window_ms"]].copy()

    sl_path = save_dir / f"shortlist_{subject}.csv"
    shortlist.to_csv(sl_path, index=False)

    return df, shortlist, lb_path, sl_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True, help="Subject ID, e.g., AB01")
    ap.add_argument("--out_root", default=str(DEFAULT_OUT_ROOT), help="Root of grid outputs")
    ap.add_argument("--save_dir", default=".", help="Directory to write leaderboard and shortlist")
    args = ap.parse_args()

    df, shortlist, lb_path, sl_path = build_subject_leaderboard(Path(args.out_root), args.subject, Path(args.save_dir))
    print(f"[OK] subject leaderboard → {lb_path} (rows={len(df)})")
    print(f"[OK] shortlist → {sl_path} (rows={len(shortlist)})")


if __name__ == "__main__":
    main()
