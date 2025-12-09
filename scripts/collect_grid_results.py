import json, os, re, csv
from pathlib import Path

ROOT = Path("out_grid")
rows = []

# Regex to capture trial, sensor, model, window from folder name
pat = re.compile(r"(?P<trial>AB\d+_.+?)_(?P<sensor>rpelvis|lpelvis|rshank|lshank|rpthigh|lpthigh|rathigh|lathigh)_(?P<model>rf|hgb|ridge)_w(?P<win>\d+)")

for d in ROOT.iterdir():
    if not d.is_dir(): 
        continue
    m = pat.match(d.name)
    if not m:
        # skip unknown folders (e.g., stray files)
        continue
    metrics_path = d / "eval" / "metrics_eval.json"
    pred_path = d / "predicted_fz.csv"
    if not metrics_path.exists():
        print(f"[WARN] Missing metrics: {d}")
        continue
    try:
        metrics = json.loads(metrics_path.read_text())
        rmse_pct = metrics.get("rmse_%BW", None)
        mae_pct = metrics.get("mae_%BW", None)
        # Pearson r was printed by train, not saved; infer from eval_compare plot not trivial.
        # If you want r, we can recompute later from true/pred CSVs.
        rows.append({
            "trial": m["trial"],
            "sensor": m["sensor"],
            "model": m["model"],
            "window_ms": int(m["win"]),
            "rmse_%BW": rmse_pct,
            "mae_%BW": mae_pct,
            "predicted_csv": str(pred_path),
            "metrics_json": str(metrics_path),
        })
    except Exception as e:
        print(f"[ERR] {d}: {e}")

rows.sort(key=lambda r: (r["trial"], r["rmse_%BW"] if r["rmse_%BW"] is not None else 1e9))

out_csv = Path("out_grid_leaderboard.csv")
with out_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

print(f"[OK] Wrote {out_csv} with {len(rows)} rows.")
# Print top-5 overall
print("\nTop-5 (lowest RMSE %BW):")
for r in sorted(rows, key=lambda r: r["rmse_%BW"] or 1e9)[:5]:
    print(f"  {r['trial']:40s} {r['sensor']:8s} {r['model']:6s} w={r['window_ms']:3d}  RMSE%={r['rmse_%BW']:.2f}  -> {r['predicted_csv']}")
