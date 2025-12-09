# scripts/collect_leaderboard_dual.py
import argparse, json, os, re
from pathlib import Path
import pandas as pd
import yaml

PATTERN = re.compile(r"^(?P<subject>AB\d+?)_(?P<trial>.+)_(?P<sensors>[A-Za-z0-9\-]+)_(?P<model>ridge|rf|hgb|xgb|stack|cnn1d)_w(?P<window>\d+)$", re.IGNORECASE)


def parse_combo(stem: str):
    m = PATTERN.match(stem)
    if not m:
        return None
    g = m.groupdict()
    g["window_ms"] = int(g.pop("window"))
    g["model_kind"] = g.pop("model").lower()
    g["subject"] = g["subject"].upper()
    return g


def load_dataset_bw(dataset_cfg: Path, subject: str) -> float | None:
    try:
        with open(dataset_cfg, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception:
        return None
    # Prefer subject_masses if present
    masses = cfg.get("subject_masses", {}) or {}
    if subject in masses:
        try:
            return float(masses[subject])
        except Exception:
            pass
    # Else default_bw_kg
    try:
        return float(cfg.get("default_bw_kg"))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="Collect leaderboard rows from out_root eval/metrics_eval.json")
    ap.add_argument("--out_root", default="E:/safestride/out_grid")
    ap.add_argument("--dataset_cfg", default="configs/dataset.yaml")
    ap.add_argument("--subject", default="AB01")
    ap.add_argument("--restrict_sensors", default=None, help="Comma-separated list; if set, keep only those sensors")
    ap.add_argument("--save_csv", default="out_grid_leaderboard_AB01.csv")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    hits = list(out_root.glob("**/eval/metrics_eval.json"))
    rows = []
    restrict = set(s.strip().lower() for s in (args.restrict_sensors or "").split(",") if s.strip())

    for mj in hits:
        combo_dir = mj.parent.parent  # .../<combo>/eval/metrics_eval.json -> <combo>
        info = parse_combo(combo_dir.name)
        if not info:
            continue
        if args.subject and info["subject"] != args.subject.upper():
            continue
        if restrict and info["sensors"].lower() not in restrict:
            continue
        try:
            with open(mj, "r", encoding="utf-8") as f:
                m = json.load(f)
        except Exception:
            continue
        rmse_pct = m.get("rmse_pctbw_mean")
        mae_N_mean = m.get("mae_N_mean")
        mae_pct = None
        bw_kg = load_dataset_bw(Path(args.dataset_cfg), info["subject"]) or None
        if bw_kg and isinstance(mae_N_mean, (int, float)):
            try:
                mae_pct = float(mae_N_mean) / (float(bw_kg) * 9.80665) * 100.0
            except Exception:
                mae_pct = None
        rows.append({
            "subject": info["subject"],
            # Include subject prefix in the trial field, e.g., AB01_cutting_1_left-fast
            "trial": f"{info['subject']}_{info['trial']}",
            "sensors": info["sensors"],
            "model_kind": info["model_kind"],
            "window_ms": int(info["window_ms"]),
            "rmse_%BW": float(rmse_pct) if rmse_pct is not None else None,
            "mae_%BW": float(mae_pct) if mae_pct is not None else None,
            "metrics_json_path": str(mj),
            "outdir": str(combo_dir),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["trial", "rmse_%BW"], ascending=[True, True])
    Path(args.save_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.save_csv, index=False)
    print(f"[OK] wrote {args.save_csv} rows={len(df)}")


if __name__ == "__main__":
    main()
