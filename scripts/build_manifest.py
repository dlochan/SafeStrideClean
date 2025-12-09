# scripts/build_manifest.py
import argparse, csv
from pathlib import Path
from src.config import DATA_ROOT

# which subfolders are the 3 files we need per trial
FN_IMU   = "_imu_real.csv"
FN_GRF   = "_grf.csv"
FN_FLAG  = "_activity_flag.csv"

SENSORS = ["rpelvis","lpelvis","rshank","lshank","rpthigh","lpthigh","rathigh","lathigh"]  # you can trim later

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)   # e.g. AB01
    ap.add_argument("--out_csv", default="data/manifest_AB01.csv")
    args = ap.parse_args()

    subj_dir = DATA_ROOT / args.subject
    rows = []
    for trial_dir in sorted(d for d in subj_dir.iterdir() if d.is_dir()):
        trial = trial_dir.name
        # expected filenames
        prefix = f"{args.subject}_{trial}"
        imu   = trial_dir / f"{prefix}{FN_IMU}"
        grf   = trial_dir / f"{prefix}{FN_GRF}"
        flag  = trial_dir / f"{prefix}{FN_FLAG}"
        if imu.exists() and grf.exists() and flag.exists():
            for sensor in SENSORS:
                rows.append([args.subject, trial, sensor, str(imu), str(grf), str(flag)])
        # silently skip trials that don’t have all 3 files

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject","trial","sensor","imu_in","grf_in","flag_in"])
        w.writerows(rows)
    print(f"[OK] wrote manifest: {args.out_csv} (rows={len(rows)})")

if __name__ == "__main__":
    main()
