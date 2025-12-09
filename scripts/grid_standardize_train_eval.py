# grid_standardize_train_eval.py
import argparse, subprocess, json, sys, os
from pathlib import Path
import pandas as pd

# --------- CLI ---------
ap = argparse.ArgumentParser(description="Grid search (sensor x model x window) with resume/overwrite.")
ap.add_argument("--resume",    action="store_true", help="Skip runs that already have predictions and metrics.")
ap.add_argument("--overwrite", action="store_true", help="Recompute everything even if outputs exist.")
# You can point these to any single trial you want to grid over:
ap.add_argument("--imu_in",  default=r"E:\safestride\datasets\ProcessedData\AB01\cutting_1_left-fast\AB01_cutting_1_left-fast_imu_real.csv")
ap.add_argument("--grf_in",  default=r"E:\safestride\datasets\ProcessedData\AB01\cutting_1_left-fast\AB01_cutting_1_left-fast_grf.csv")
ap.add_argument("--flag_in", default=r"E:\safestride\datasets\ProcessedData\AB01\cutting_1_left-fast\AB01_cutting_1_left-fast_activity_flag.csv")
ap.add_argument("--bw_kg",   type=float, default=78.9)
ap.add_argument("--fs",      type=int,   default=200)
ap.add_argument("--work_dir", default="data/working")
ap.add_argument("--out_root", default="out_grid")
args = ap.parse_args()

# --------- USER GRID ---------
SENSORS    = ["rpelvis","lpelvis","rshank","lshank","rpthigh","lpthigh","rathigh","lathigh"]
MODELS     = ["rf","hgb","ridge"]
WINDOW_MS  = [200, 300, 400]
RESULTS_CSV = Path(args.out_root) / "grid_results.csv"
# --------------------------------

SAFE = Path(".").resolve()

def run_is_done(outdir: str) -> bool:
    pred = os.path.join(outdir, "predicted_fz.csv")
    met  = os.path.join(outdir, "eval", "metrics_eval.json")
    return os.path.isfile(pred) and os.path.isfile(met)

def run(cmd, cwd=None):
    print("\n> ", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, shell=False)

def ensure_dir(p: Path):
    if p.suffix:  # path is a file (has extension)
        p.parent.mkdir(parents=True, exist_ok=True)
    else:
        p.mkdir(parents=True, exist_ok=True)

def main():
    work_dir = Path(args.work_dir)
    out_root = Path(args.out_root)
    ensure_dir(work_dir / "dummy.txt")
    ensure_dir(out_root)

    rows = []

    for sensor in SENSORS:
        # standardized outputs (cached)
        imu_out = work_dir / f"gt_AB01_cutting_leftfast_imu_active_{sensor}.csv"
        grf_out = work_dir / f"gt_AB01_cutting_leftfast_grf_active_{sensor}.csv"
        grf_shifted = work_dir / f"gt_AB01_cutting_leftfast_grf_active_{sensor}_shifted.csv"

        # 1) Standardize (skip if exists and --resume)
        if imu_out.exists() and grf_out.exists() and args.resume and not args.overwrite:
            print(f"[RESUME] Using cached standardized files for {sensor}")
        else:
            run([
                sys.executable, "scripts/standardize_gt_trial_active.py",
                "--imu_in",  args.imu_in,
                "--grf_in",  args.grf_in,
                "--flag_in", args.flag_in,
                "--sensor",  sensor,
                "--imu_out", str(imu_out),
                "--grf_out", str(grf_out),
            ])

        # 2) Align (skip if exists and --resume)
        if grf_shifted.exists() and args.resume and not args.overwrite:
            print(f"[RESUME] Using cached aligned GRF for {sensor}")
        else:
            run([
                sys.executable, "scripts/auto_align_shift_grf.py",
                "--imu", str(imu_out),
                "--grf_in", str(grf_out),
                "--grf_out", str(grf_shifted),
                "--fs", str(args.fs)
            ])

        # 3) Train/Eval grid
        for model in MODELS:
            for w in WINDOW_MS:
                outdir = out_root / f"AB01_cutting_leftfast_{sensor}_{model}_w{w}"
                ensure_dir(outdir)

                if args.resume and not args.overwrite and run_is_done(str(outdir)):
                    print(f"[RESUME] Skipping already complete: {outdir}")
                    continue

                # Train
                run([
                    sys.executable, "-m", "src.train",
                    "--imu_csv",  str(imu_out),
                    "--grf_path", str(grf_shifted),
                    "--grf_type", "csv",
                    "--bw_kg",    str(args.bw_kg),
                    "--fs_hint",  str(args.fs),
                    "--window_ms", str(w),
                    "--model",    model,
                    "--outdir",   str(outdir)
                ])

                # Eval
                pred_csv = outdir / "predicted_fz.csv"
                eval_dir = outdir / "eval"
                ensure_dir(eval_dir)
                run([
                    sys.executable, "-m", "src.eval_compare",
                    "--true_grf_csv", str(grf_shifted),
                    "--pred_grf_csv", str(pred_csv),
                    "--outdir",       str(eval_dir),
                    "--bw_kg",        str(args.bw_kg)
                ])

                # Append metrics row
                metrics_path = eval_dir / "metrics_eval.json"
                with open(metrics_path, "r") as f:
                    m = json.load(f)
                rows.append({
                    "sensor": sensor,
                    "model": model,
                    "window_ms": w,
                    "n_samples": m.get("n_samples"),
                    "rmse_N": m.get("rmse_N"),
                    "mae_N":  m.get("mae_N"),
                    "rmse_%BW": m.get("rmse_%BW"),
                    "mae_%BW":  m.get("mae_%BW"),
                    "outdir":   str(outdir)
                })

    # 4) Save summary table
    if rows:
        df = pd.DataFrame(rows).sort_values(["rmse_%BW"])
        ensure_dir(RESULTS_CSV)
        df.to_csv(RESULTS_CSV, index=False)
        print("\nSaved summary:", RESULTS_CSV)
        print(df.head(12).to_string(index=False))
    else:
        print("[INFO] Nothing new to run (resume found everything complete).")

if __name__ == "__main__":
    main()
