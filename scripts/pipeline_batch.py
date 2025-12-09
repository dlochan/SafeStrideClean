# scripts/pipeline_batch.py
import argparse, csv, json, subprocess, sys, time
from pathlib import Path
from src.config import BW_BY_SUBJECT, OUT_ROOT

def sh(cmd):
    print("\n>"," ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode

def already_done(subject, trial, sensor):
    # if we already have a best.json for this stem, skip
    stem = f"{subject}_{trial}_{sensor}"
    return (OUT_ROOT / "best" / f"{stem}_best.json").exists()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    with open(args.manifest, newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit: rows = rows[:args.limit]

    for i, r in enumerate(rows, 1):
        subj   = r["subject"]
        trial  = r["trial"]
        sensor = r["sensor"]
        if already_done(subj, trial, sensor):
            print(f"[skip] best exists for {subj}/{trial}/{sensor}")
            continue
        bw = BW_BY_SUBJECT.get(subj)
        if bw is None:
            print(f"[warn] no bodyweight for {subj}; skipping"); continue
        cmd = [
            sys.executable, "scripts/pipeline_one_trial.py",
            "--subject", subj,
            "--trial", trial,
            "--sensor", sensor,
            "--imu_in", r["imu_in"],
            "--grf_in", r["grf_in"],
            "--flag_in", r["flag_in"],
            "--bw_kg", str(bw),
        ]
        rc = sh(cmd)
        if rc != 0:
            print(f"[ERROR] {subj}/{trial}/{sensor} rc={rc} (continuing)")
        # tiny pause to be nice to disk
        time.sleep(0.2)

if __name__ == "__main__":
    main()
