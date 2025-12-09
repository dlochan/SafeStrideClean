# scripts/pipeline_one_trial.py
import argparse, json, subprocess, sys
from pathlib import Path
from src.config import WORK_ROOT, OUT_ROOT, FS_HZ

def sh(cmd_list):
    print("\n> "," ".join(cmd_list), flush=True)
    cp = subprocess.run(cmd_list)
    if cp.returncode != 0:
        raise SystemExit(cp.returncode)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--sensor", required=True)
    ap.add_argument("--imu_in", required=True)
    ap.add_argument("--grf_in", required=True)
    ap.add_argument("--flag_in", required=True)
    ap.add_argument("--bw_kg", required=True)
    ap.add_argument("--windows_ms", default="200,300,400")
    ap.add_argument("--models", default="rf,hgb,ridge")
    args = ap.parse_args()

    # working files (unique path per trial+sensor)
    stem = f"{args.subject}_{args.trial}_{args.sensor}"
    imu_work = WORK_ROOT / f"{stem}_imu_active.csv"
    grf_work = WORK_ROOT / f"{stem}_grf_active.csv"
    grf_shift= WORK_ROOT / f"{stem}_grf_active_shifted.csv"

    # 1) standardize & crop to active
    sh([
        sys.executable, "scripts/standardize_gt_trial_active.py",
        "--imu_in",  args.imu_in,
        "--grf_in",  args.grf_in,
        "--flag_in", args.flag_in,
        "--sensor",  args.sensor,
        "--imu_out", str(imu_work),
        "--grf_out", str(grf_work),
    ])

    # 2) auto-align & write shifted GRF
    sh([
        sys.executable, "scripts/auto_align_shift_grf.py",
        "--imu", str(imu_work),
        "--grf_in", str(grf_work),
        "--grf_out", str(grf_shift),
        "--fs", str(FS_HZ),
    ])

    # 3) run grid of models/windows
    best = None
    for model in args.models.split(","):
        for w in map(int, args.windows_ms.split(",")):
            outdir = OUT_ROOT / f"{args.subject}_{args.trial}_{args.sensor}_{model}_w{w}"
            sh([
                sys.executable, "-m", "src.train",
                "--imu_csv", str(imu_work),
                "--grf_path", str(grf_shift),
                "--grf_type", "csv",
                "--bw_kg", str(args.bw_kg),
                "--fs_hint", str(FS_HZ),
                "--window_ms", str(w),
                "--model", model,
                "--outdir", str(outdir),
            ])
            # evaluate + read metrics for picking "best"
            eval_dir = outdir / "eval"
            sh([
                sys.executable, "-m", "src.eval_compare",
                "--true_grf_csv", str(grf_shift),
                "--pred_grf_csv", str(outdir / "predicted_fz.csv"),
                "--outdir", str(eval_dir),
                "--bw_kg", str(args.bw_kg),
            ])
            mpath = eval_dir / "metrics_eval.json"
            with open(mpath) as f: m = json.load(f)
            key = (m.get("rmse_%BW", 1e9), -m.get("rmse_N", 0))  # mainly min %BW
            if (best is None) or (key < best[0]):
                best = (key, {"model":model,"window_ms":w,"metrics":m,"outdir":str(outdir)})
    # write a small best.json beside working files
    if best:
        (OUT_ROOT / "best").mkdir(exist_ok=True, parents=True)
        bfile = OUT_ROOT / "best" / f"{stem}_best.json"
        with open(bfile, "w") as f: json.dump(best[1], f, indent=2)
        print("[BEST]", bfile)

if __name__ == "__main__":
    main()
