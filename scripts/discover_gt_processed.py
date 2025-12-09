import argparse, os, glob, pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help=r"e.g. E:\safestride\datasets\ProcessedData")
    ap.add_argument("--save_csv", default=r"E:\safestride\manifests\gt_processed_manifest.csv")
    args = ap.parse_args()

    rows=[]
    for subject_dir in sorted(glob.glob(os.path.join(args.root, "AB*"))):
        subj = os.path.basename(subject_dir)
        for task_dir in sorted(glob.glob(os.path.join(subject_dir, "*"))):
            if not os.path.isdir(task_dir): 
                continue
            # find three files we need
            base = os.path.basename(task_dir)  # e.g., cutting_1_left-fast
            imu = os.path.join(task_dir, f"{subj}_{base}_imu_real.csv")
            grf = os.path.join(task_dir, f"{subj}_{base}_grf.csv")
            flag = os.path.join(task_dir, f"{subj}_{base}_activity_flag.csv")
            if all(os.path.exists(p) for p in (imu,grf,flag)):
                rows.append({
                    "subject": subj,
                    "trial": f"{subj}_{base}",
                    "imu_real": imu,
                    "grf": grf,
                    "activity_flag": flag
                })
    os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(args.save_csv, index=False)
    print(f"[OK] wrote manifest with {len(rows)} trials to {args.save_csv}")

if __name__ == "__main__":
    main()
