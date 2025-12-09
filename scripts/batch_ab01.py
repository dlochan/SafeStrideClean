# scripts/batch_ab01.py
# Runs full pipeline for ALL AB01 trials, writing outputs to E:
# Requirements: your existing scripts + src.* modules already working.

import os
import json
import subprocess
from pathlib import Path
from datetime import datetime

# ------------------------
# CONFIG (edit if needed)
# ------------------------
DATA_ROOT = Path(r"E:\safestride\datasets\ProcessedData")  # raw processed dataset root
SUBJECT    = "AB01"                                        # this script does AB01 only
WORK_ROOT  = Path(r"E:\safestride\working")                # intermediates (standardized CSVs)
OUT_ROOT   = Path(r"E:\safestride\out_grid")               # model outputs + evals
LOG_DIR    = Path(r"E:\safestride\logs")                   # simple text logs

FS_HZ      = 200
BW_KG      = 78.9  # AB01 subject mass from dataset readme
SENSORS    = ["rpelvis","lpelvis","rshank","lshank","rpthigh","lpthigh","rathigh","lathigh"]
MODELS     = ["rf","hgb","ridge"]                         # you can trim to speed up
WINDOW_MS  = [200, 300, 400]

# Skip training/eval if prediction file already exists to allow resuming
SKIP_IF_DONE = True

# ------------------------
# Helpers
# ------------------------
def run(cmd:list, cwd:Path=None) -> int:
    """Run a command (list form) and stream output."""
    print("> ", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def discover_trials(subject:str):
    """Return list of (trial_name, trial_dir) for AB01."""
    subj_dir = DATA_ROOT / subject
    trials = []
    for item in sorted(subj_dir.iterdir()):
        if not item.is_dir():
            continue
        # only accept if it has imu_real.csv + grf.csv + activity_flag.csv
        base = f"{subject}_{item.name}"
        imu_csv  = item / f"{base}_imu_real.csv"
        grf_csv  = item / f"{base}_grf.csv"
        flag_csv = item / f"{base}_activity_flag.csv"
        if imu_csv.exists() and grf_csv.exists() and flag_csv.exists():
            trials.append((item.name, item))
    return trials

def write_leaderboard(rows, out_csv:Path):
    import csv
    fieldnames = [
        "trial","sensor","model","window_ms",
        "rmse_%BW","mae_%BW",
        "predicted_metrics_json"
    ]
    ensure_dirs(out_csv.parent)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[OK] wrote leaderboard: {out_csv}")

# ------------------------
# Main
# ------------------------
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    ensure_dirs(WORK_ROOT, OUT_ROOT, LOG_DIR)

    trials = discover_trials(SUBJECT)
    if not trials:
        print(f"[ERR] No trials found under {DATA_ROOT / SUBJECT}")
        return
    print(f"[INFO] Found {len(trials)} trials for {SUBJECT}")

    leaderboard_rows = []
    log_path = LOG_DIR / f"batch_ab01_{ts}.log"
    with open(log_path, "w") as log:
        log.write(f"Batch run {ts} for {SUBJECT}\n")
        log.write(f"Trials: {len(trials)}\n")

    # Loop trials
    for trial_name, trial_dir in trials:
        base = f"{SUBJECT}_{trial_name}"
        imu_in  = trial_dir / f"{base}_imu_real.csv"
        grf_in  = trial_dir / f"{base}_grf.csv"
        flag_in = trial_dir / f"{base}_activity_flag.csv"

        print(f"\n=== Trial: {trial_name} ===")
        # Each sensor
        for sensor in SENSORS:
            imu_out = WORK_ROOT / f"gt_{SUBJECT}_{trial_name}_imu_active_{sensor}.csv"
            grf_out = WORK_ROOT / f"gt_{SUBJECT}_{trial_name}_grf_active_{sensor}.csv"
            grf_shifted = WORK_ROOT / f"gt_{SUBJECT}_{trial_name}_grf_active_{sensor}_shifted.csv"

            # 1) Standardize (active only, sensor slice)
            rc = run([
                "python","scripts/standardize_gt_trial_active.py",
                "--imu_in", str(imu_in),
                "--grf_in", str(grf_in),
                "--flag_in", str(flag_in),
                "--sensor", sensor,
                "--imu_out", str(imu_out),
                "--grf_out", str(grf_out)
            ])
            if rc != 0:
                print(f"[WARN] standardize failed for {trial_name} {sensor}, skipping.")
                continue

            # 2) Align GRF to IMU (auto)
            rc = run([
                "python","scripts/auto_align_shift_grf.py",
                "--imu", str(imu_out),
                "--grf_in", str(grf_out),
                "--grf_out", str(grf_shifted),
                "--fs", str(FS_HZ)
            ])
            if rc != 0:
                print(f"[WARN] align failed for {trial_name} {sensor}, skipping.")
                continue

            # 3) Train + Eval across models/windows
            for model in MODELS:
                for win in WINDOW_MS:
                    outdir = OUT_ROOT / f"{SUBJECT}_{trial_name}_{sensor}_{model}_w{win}"
                    pred_csv = outdir / "predicted_fz.csv"
                    if SKIP_IF_DONE and pred_csv.exists():
                        # Still collect metrics to leaderboard (eval already wrote metrics_eval.json)
                        metrics_json = outdir / "eval" / "metrics_eval.json"
                        if metrics_json.exists():
                            try:
                                with open(metrics_json) as f:
                                    mj = json.load(f)
                                leaderboard_rows.append({
                                    "trial": trial_name,
                                    "sensor": sensor,
                                    "model": model,
                                    "window_ms": win,
                                    "rmse_%BW": mj.get("rmse_%BW", ""),
                                    "mae_%BW": mj.get("mae_%BW", ""),
                                    "predicted_metrics_json": str(metrics_json)
                                })
                            except Exception:
                                pass
                        continue

                    ensure_dirs(outdir)

                    # Train
                    rc = run([
                        "python","-m","src.train",
                        "--imu_csv", str(imu_out),
                        "--grf_path", str(grf_shifted),
                        "--grf_type", "csv",
                        "--bw_kg", f"{BW_KG}",
                        "--fs_hint", f"{FS_HZ}",
                        "--window_ms", f"{win}",
                        "--model", model,
                        "--outdir", str(outdir)
                    ])
                    if rc != 0:
                        print(f"[WARN] train failed {trial_name} {sensor} {model} w{win}")
                        continue

                    # Eval
                    eval_outdir = outdir / "eval"
                    rc = run([
                        "python","-m","src.eval_compare",
                        "--true_grf_csv", str(grf_shifted),
                        "--pred_grf_csv", str(pred_csv),
                        "--outdir", str(eval_outdir),
                        "--bw_kg", f"{BW_KG}"
                    ])
                    if rc != 0:
                        print(f"[WARN] eval failed {trial_name} {sensor} {model} w{win}")
                        continue

                    # Read metrics for leaderboard
                    metrics_json = eval_outdir / "metrics_eval.json"
                    try:
                        with open(metrics_json) as f:
                            mj = json.load(f)
                        leaderboard_rows.append({
                            "trial": trial_name,
                            "sensor": sensor,
                            "model": model,
                            "window_ms": win,
                            "rmse_%BW": mj.get("rmse_%BW", ""),
                            "mae_%BW": mj.get("mae_%BW", ""),
                            "predicted_metrics_json": str(metrics_json)
                        })
                    except Exception as e:
                        print(f"[WARN] could not read metrics: {e}")

    # Write leaderboard
    lb_csv = OUT_ROOT / "out_grid_leaderboard_AB01.csv"
    write_leaderboard(leaderboard_rows, lb_csv)

    print("\n[DONE] Batch for AB01 complete.")
    print(f"Leaderboard: {lb_csv}")
    print(f"Logs: {log_path}")

if __name__ == "__main__":
    main()
