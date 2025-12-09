# scripts/make_shortlist.py
import argparse, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--in_csv", required=True)    # e.g. top_configs_AB01_kneeonly.csv
ap.add_argument("--out_csv", required=True)   # e.g. shortlist_AB01_kneeonly.csv
ap.add_argument("--per_trial", type=int, default=1, help="Top N rows per trial (default 1)")
args = ap.parse_args()

df = pd.read_csv(args.in_csv)

# sanity: needed columns
need = {"trial","sensor","model","window_ms","rmse_%BW"}
missing = need - set(df.columns)
if missing:
    raise SystemExit(f"Missing columns in {args.in_csv}: {missing}")

# keep the best per trial (lowest RMSE)
df_sorted = df.sort_values(["trial","rmse_%BW"], ascending=[True, True])
shortlist = df_sorted.groupby("trial", as_index=False).head(args.per_trial)

# keep only the columns batch_subjects.py requires
shortlist = shortlist[["trial","sensor","model","window_ms"]].copy()

shortlist.to_csv(args.out_csv, index=False)
print(f"[OK] wrote {args.out_csv} with {len(shortlist)} rows")

