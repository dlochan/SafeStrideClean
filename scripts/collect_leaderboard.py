# scripts/collect_leaderboard.py
import json, csv
from pathlib import Path
from src.config import OUT_ROOT

def main():
    rows = []
    for best in (OUT_ROOT / "best").glob("*_best.json"):
        stem = best.stem.replace("_best","")  # SUBJECT_TRIAL_SENSOR
        with open(best) as f: info = json.load(f)
        m = info["metrics"]
        subj, trial, sensor = stem.split("_", 2)
        rows.append({
            "subject": subj,
            "trial": trial,
            "sensor": sensor,
            "model": info["model"],
            "window_ms": info["window_ms"],
            "rmse_%BW": round(m.get("rmse_%BW", float("nan")), 3),
            "mae_%BW": round(m.get("mae_%BW", float("nan")), 3),
            "outdir": info["outdir"]
        })
    rows.sort(key=lambda r: (r["subject"], r["trial"], r["rmse_%BW"]))
    out_csv = OUT_ROOT / "leaderboard.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["subject","trial","sensor","model","window_ms","rmse_%BW","mae_%BW","outdir"])
        w.writeheader(); w.writerows(rows)
    print(f"[OK] leaderboard → {out_csv} (rows={len(rows)})")

if __name__ == "__main__":
    main()
