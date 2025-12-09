# scripts/batch_subjects.py
import argparse, os, subprocess, pandas as pd, pathlib, json, sys

SAFE = r"C:\Users\locha\Documents\safestride"  # project root

def run(cmd):
    print("\n> ", " ".join(cmd))
    subprocess.run(cmd, check=True)

def run_is_done(outdir: str) -> bool:
    pred = os.path.join(outdir, "predicted_fz.csv")
    met  = os.path.join(outdir, "eval", "metrics_eval.json")
    return os.path.isfile(pred) and os.path.isfile(met)

def try_standardize(imu_in, grf_in, flag_in, sensor, imu_out, grf_out):
    try:
        run([
            "python", os.path.join(SAFE,"scripts","standardize_gt_trial_active.py"),
            "--imu_in", imu_in,
            "--grf_in", grf_in,
            "--flag_in", flag_in,
            "--sensor", sensor,
            "--imu_out", imu_out,
            "--grf_out", grf_out
        ])
        return True, None
    except subprocess.CalledProcessError as e:
        return False, e

def try_align(imu_out, grf_out, grf_shift, fs):
    try:
        run([
            "python", os.path.join(SAFE,"scripts","auto_align_shift_grf.py"),
            "--imu", imu_out,
            "--grf_in", grf_out,
            "--grf_out", grf_shift,
            "--fs", str(fs)
        ])
        return True, None
    except subprocess.CalledProcessError as e:
        return False, e

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--shortlist_csv", required=True)  # from AB01
    ap.add_argument("--subjects", nargs="+", required=True)  # e.g. AB02 AB03 ...
    ap.add_argument("--out_root", default=r"E:\safestride\out_grid")
    ap.add_argument("--sensor_fallback", default="rshank")   # <-- default updated to r/l shank
    ap.add_argument("--resume",    action="store_true", help="Skip runs already complete; reuse cached standardization.")
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if outputs exist.")
    ap.add_argument("--fs", type=int, default=200)
    ap.add_argument("--bw_kg", type=float, default=78.9)
    args = ap.parse_args()

    manifest = pd.read_csv(args.manifest_csv)
    picks = pd.read_csv(args.shortlist_csv)  # columns: trial,sensor,model,window_ms,...

    for subj in args.subjects:
        sub_rows = manifest[manifest["subject"]==subj].copy()
        print(f"\n=== Subject {subj} | {len(sub_rows)} trials ===")

        for _, row in sub_rows.iterrows():
            trial_name = row["trial"]                      # e.g., AB02_cutting_1_left-fast
            task_key = "_".join(trial_name.split("_")[1:]) # cutting_1_left-fast
            ab01_key = f"AB01_{task_key}"

            # shortlist rows that match this AB01 task
            S = picks[picks["trial"]==ab01_key]
            if S.empty:
                print(f"[WARN] no shortlist for {ab01_key}, using fallback sensor {args.sensor_fallback} with RF w300")
                S = pd.DataFrame([{"sensor": args.sensor_fallback, "model":"rf", "window_ms":300}])

            # working paths on C:
            work_imu = os.path.join(SAFE, "data", "working", f"{trial_name}_imu_active.csv")
            work_grf = os.path.join(SAFE, "data", "working", f"{trial_name}_grf_active.csv")
            work_grf_shift = os.path.join(SAFE, "data", "working", f"{trial_name}_grf_active_shifted.csv")

            # choose the sensor of first row for standardization
            sensor_std = S.iloc[0]["sensor"]

            # STANDARDIZE (skip if cached and --resume)
            if (os.path.isfile(work_imu) and os.path.isfile(work_grf) and args.resume and not args.overwrite):
                print(f"[RESUME] Using cached standardized files for {trial_name}")
                std_ok = True
            else:
                std_ok, std_err = try_standardize(
                    row["imu_real"], row["grf"], row["activity_flag"],
                    sensor_std, work_imu, work_grf
                )
                if not std_ok:
                    print(f"[SKIP] standardize failed for {trial_name} ({std_err}); skipping trial.")
                    continue

            # ALIGN (skip if cached and --resume)
            if os.path.isfile(work_grf_shift) and args.resume and not args.overwrite:
                print(f"[RESUME] Using cached aligned GRF for {trial_name}")
                align_ok = True
            else:
                align_ok, align_err = try_align(work_imu, work_grf, work_grf_shift, args.fs)
                if not align_ok:
                    print(f"[SKIP] align failed for {trial_name} ({align_err}); skipping trial.")
                    continue

            # loop each shortlisted config
            for _, cfg in S.iterrows():
                sensor = cfg["sensor"]
                model  = cfg["model"]
                w      = int(cfg["window_ms"])
                outdir = os.path.join(args.out_root, f"{trial_name}_{sensor}_{model}_w{w}")

                if args.resume and not args.overwrite and run_is_done(outdir):
                    print(f"[RESUME] Skipping already complete: {outdir}")
                    continue

                # TRAIN
                try:
                    run([
                        "python", "-m", "src.train",
                        "--imu_csv", work_imu,
                        "--grf_path", work_grf_shift,
                        "--grf_type", "csv",
                        "--bw_kg", str(args.bw_kg),
                        "--fs_hint", str(args.fs),
                        "--window_ms", str(w),
                        "--model", model,
                        "--outdir", outdir
                    ])
                except subprocess.CalledProcessError as e:
                    print(f"[SKIP] training failed for {outdir}: {e}; skipping this config.")
                    continue

                # EVAL
                try:
                    run([
                        "python", "-m", "src.eval_compare",
                        "--true_grf_csv", work_grf_shift,
                        "--pred_grf_csv", os.path.join(outdir,"predicted_fz.csv"),
                        "--outdir", os.path.join(outdir,"eval"),
                        "--bw_kg", str(args.bw_kg)
                    ])
                except subprocess.CalledProcessError as e:
                    print(f"[SKIP] eval failed for {outdir}: {e}; skipping eval.")
                    continue

if __name__ == "__main__":
    main()
