# scripts/make_shortlist_dual.py
import argparse
import pandas as pd
from pathlib import Path


def is_dual(sensors: str) -> bool:
    # Consider dual if it includes a hyphen (e.g., 'thigh-shank', 'lpthigh-lshank')
    return isinstance(sensors, str) and ("-" in sensors)


def main():
    ap = argparse.ArgumentParser(description="Build a shortlist from a leaderboard, preferring dual sensors")
    ap.add_argument("--leaderboard", required=True, help="CSV with columns: subject,trial,sensors,model_kind,window_ms,rmse_%BW,mae_%BW,metrics_json_path,outdir")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--prefer_dual", action="store_true", help="If set, keep only dual-sensor rows")
    ap.add_argument("--top_per_model", type=int, default=1, help="Top N per (trial, model_kind)")
    ap.add_argument("--top_per_window", type=int, default=1, help="Top N per (trial, window_ms)")
    args = ap.parse_args()

    df = pd.read_csv(args.leaderboard)
    need = {"subject","trial","sensors","model_kind","window_ms","rmse_%BW"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"Missing columns in leaderboard: {miss}")

    # Prefer dual sensors if requested
    if args.prefer_dual:
        before = len(df)
        df = df[df["sensors"].apply(is_dual)].copy()
        print(f"[INFO] prefer_dual: kept {len(df)}/{before} rows with dual sensors")

    # Drop NaN rmse rows
    df = df.dropna(subset=["rmse_%BW"]).copy()

    # Sort by rmse ascending
    df = df.sort_values(["trial","rmse_%BW"], ascending=[True, True])

    picks = []
    # Best per (trial, model_kind)
    if args.top_per_model > 0:
        per_model = df.groupby(["trial","model_kind"], as_index=False).head(args.top_per_model)
        picks.append(per_model)
    # Best per (trial, window_ms)
    if args.top_per_window > 0:
        per_window = df.groupby(["trial","window_ms"], as_index=False).head(args.top_per_window)
        picks.append(per_window)

    if picks:
        out_df = pd.concat(picks, ignore_index=True).drop_duplicates()
    else:
        out_df = df.copy()

    out_df = out_df.sort_values(["trial","model_kind","window_ms","rmse_%BW"]).reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[OK] wrote {out_path} rows={len(out_df)}")


if __name__ == "__main__":
    main()
