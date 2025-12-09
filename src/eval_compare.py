import argparse
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

def pick(df, names):
    for n in names:
        if n in df.columns:
            return n
    raise KeyError(f"None of {names} found in columns: {list(df.columns)}")

def load_true_grf(path, bw_kg=None):
    df = pd.read_csv(path)
    # Accept time or time_s; rename to time_s
    tcol = "time_s" if "time_s" in df.columns else ("time" if "time" in df.columns else None)
    if tcol is None:
        raise KeyError(f"No time/time_s column in {path}")
    if tcol != "time_s":
        df = df.rename(columns={tcol: "time_s"})
    # Try Newtons first
    candN = ["Fz_N","fz_n","Fz","fz","vertical","vertical_N","Fz_total","FzN"]
    for c in candN:
        if c in df.columns:
            return df[["time_s", c]].rename(columns={c: "Fz_N"}).sort_values("time_s").reset_index(drop=True)
    # Fall back to %BW
    candBW = ["Fz_%BW","Fz_percentBW","Fz_BW","fz_bw","fz_%bw","vertical_%BW"]
    for c in candBW:
        if c in df.columns:
            if bw_kg is None:
                raise KeyError(f"Found {c} in true GRF (%BW) but --bw_kg not provided to convert to Newtons.")
            bwN = float(bw_kg) * 9.80665
            out = pd.DataFrame({
                "time_s": df["time_s"],
                "Fz_N": df[c] * bwN / 100.0
            })
            return out.sort_values("time_s").reset_index(drop=True)
    raise KeyError(f"No recognizable true GRF column in {path}. Found: {list(df.columns)}")

def load_pred(path, bw_kg=None):
    df = pd.read_csv(path)
    # normalize time column
    tcol = "time_s" if "time_s" in df.columns else ("time" if "time" in df.columns else None)
    if tcol is None:
        raise KeyError(f"No time/time_s column in {path}")
    if tcol != "time_s":
        df = df.rename(columns={tcol: "time_s"})

    # try Newtons first
    candN = ["Fz_pred_N", "Fz_N","fz_n","Fz","fz","Fz_pred","fz_pred","pred","prediction"]
    for c in candN:
        if c in df.columns:
            return df[["time_s", c]].rename(columns={c: "Fz_N"}).sort_values("time_s").reset_index(drop=True)

    # fall back to %BW
    candBW = ["Fz_pred_BW","fz_pred_bw","Fz_BW","fz_bw","pred_bw","prediction_bw","Fz_percentBW","Fz_%BW"]
    for c in candBW:
        if c in df.columns:
            if bw_kg is None:
                raise KeyError(f"Found {c} (%BW) but --bw_kg not provided to convert to Newtons.")
            bwN = float(bw_kg) * 9.80665
            out = pd.DataFrame({
                "time_s": df["time_s"],
                "Fz_N": df[c] * bwN / 100.0
            })
            return out.sort_values("time_s").reset_index(drop=True)

    raise KeyError(f"No recognizable prediction column in {path}. Found: {list(df.columns)}")


def align_on_time(true_df, pred_df):
    # Inner join on time; if your times are slightly off, we could do
    # a nearest-merge, but start simple.
    merged = true_df.merge(pred_df, on="time_s", how="inner", suffixes=("_true","_pred"))
    # If too few rows, fall back to nearest 5 ms
    if len(merged) < 10:
        # nearest align
        pred_df = pred_df.set_index("time_s")
        vals = []
        for t, z in zip(true_df["time_s"], true_df["Fz_N"]):
            idx = pred_df.index.get_indexer([t], method="nearest")
            vals.append(pred_df.iloc[idx[0]]["Fz_N"])
        merged = pd.DataFrame({"time_s": true_df["time_s"], "Fz_N_true": true_df["Fz_N"], "Fz_N_pred": vals})
    return merged

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--true_grf_csv", required=True)
    ap.add_argument("--pred_grf_csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--bw_kg", type=float, default=None)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    true_df = load_true_grf(args.true_grf_csv, bw_kg=args.bw_kg)
    pred_df = load_pred(args.pred_grf_csv, bw_kg=args.bw_kg)
    m = align_on_time(true_df, pred_df)

    # RMSE in Newtons
    rmse_N = float(np.sqrt(np.mean((m["Fz_N_true"] - m["Fz_N_pred"])**2)))
    mae_N  = float(np.mean(np.abs(m["Fz_N_true"] - m["Fz_N_pred"])))

    out = {"rmse_N": rmse_N, "mae_N": mae_N, "n_samples": int(len(m))}
    # Percent bodyweight metrics if provided
    if args.bw_kg:
        g = 9.80665
        bwN = args.bw_kg * g
        out["rmse_%BW"] = float(100.0 * rmse_N / bwN)
        out["mae_%BW"]  = float(100.0 * mae_N  / bwN)

    with open(os.path.join(args.outdir, "metrics_eval.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Quick plot
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(m["time_s"], m["Fz_N_true"], label="True Fz")
        plt.plot(m["time_s"], m["Fz_N_pred"], label="Pred Fz", alpha=0.8)
        plt.xlabel("Time (s)")
        plt.ylabel("Vertical GRF (N)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "fz_overlay.png"), dpi=150)
        plt.close()
    except Exception as e:
        print("[WARN] Plotting failed:", e)

    print("Saved:", os.path.join(args.outdir, "metrics_eval.json"))

if __name__ == "__main__":
    main()
