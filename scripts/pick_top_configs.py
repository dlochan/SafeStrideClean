import argparse, pandas as pd, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--leaderboard_csv", required=True)
    ap.add_argument("--per_trial", type=int, default=2,
                    help="pick top N configs per trial (lowest rmse_%BW)")
    ap.add_argument("--save_csv", default="top_configs_AB01.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.leaderboard_csv)
    # drop NaNs, sort by rmse
    df = df.dropna(subset=["rmse_%BW"]).sort_values(["trial","rmse_%BW"])
    picks = df.groupby("trial", as_index=False).head(args.per_trial)
    picks.to_csv(args.save_csv, index=False)
    print(f"[OK] wrote {args.save_csv} with {len(picks)} rows")

if __name__ == "__main__":
    main()
