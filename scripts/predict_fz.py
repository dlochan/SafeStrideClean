import argparse
from pathlib import Path
import subprocess
import pandas as pd
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = REPO_ROOT / '.venv' / 'Scripts' / 'python.exe'

ap = argparse.ArgumentParser(description='Wrapper: predict Fz then add uncertainty and clinical scores (single trial)')
ap.add_argument('--imu_csv', required=True)
ap.add_argument('--model_pkl', required=True)
ap.add_argument('--bw_kg', type=float, required=True)
ap.add_argument('--window_ms', type=int, default=300)
ap.add_argument('--trial', required=True)
ap.add_argument('--task', required=True)
ap.add_argument('--outdir', required=True)
args = ap.parse_args()

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)

# 1) Predict GRF to predicted_fz.csv with adaptive window fallback
win_try = [int(args.window_ms)]
for w in [300, 200, 100]:
    if w not in win_try:
        win_try.append(w)

last_err = None
for w in win_try:
    try:
        print(f"[predict_fz] Trying window_ms={w}")
        rc = subprocess.run([str(PY), '-m', 'src.predict_grf',
                             '--imu_csv', args.imu_csv,
                             '--model_pkl', args.model_pkl,
                             '--fs_hint', '200',
                             '--window_ms', str(w),
                             '--bw_kg', str(args.bw_kg),
                             '--out_csv', str(outdir/'predicted_fz.csv')],
                            check=False)
        if rc.returncode == 0:
            args.window_ms = w
            break
        else:
            last_err = f"predict_grf failed (code={rc.returncode}) for window_ms={w}"
    except Exception as e:
        last_err = str(e)

if (outdir/"predicted_fz.csv").exists() is False:
    raise SystemExit(last_err or "predict_grf failed for all window sizes")

# 1a) Write/merge pi_meta.json
meta_path = outdir/'pi_meta.json'
meta = {'bw_kg': float(args.bw_kg), 'window_ms': int(args.window_ms)}
try:
    if meta_path.exists():
        try:
            existing = json.loads(meta_path.read_text(encoding='utf-8'))
            if isinstance(existing, dict):
                existing.update(meta)
                meta = existing
        except Exception:
            pass
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
except Exception:
    pass

# 1b) Ensure Fz_%BW column is consistent with Fz_N (overwrite if needed)
pred_csv = outdir/'predicted_fz.csv'
if pred_csv.exists():
    try:
        d = pd.read_csv(pred_csv)
        if 'Fz_N' in d.columns:
            bwN = float(args.bw_kg) * 9.80665
            # recompute from Fz_N
            d['Fz_%BW'] = pd.to_numeric(d['Fz_N'], errors='coerce') / bwN * 100.0
            d.to_csv(pred_csv, index=False)
    except Exception:
        pass

# 2) Add uncertainty (requires baseline leaderboard for per-trial RMSE; optional)
try:
    subprocess.run([str(PY), str(REPO_ROOT/'tools'/'add_uncertainty.py'),
                    '--trial', args.trial, '--outdir', str(outdir)], check=False)
except Exception:
    pass

# 3) Clinical scores for this trial
subprocess.run([str(PY), str(REPO_ROOT/'tools'/'build_clinical_scores.py'),
                '--trial', args.trial, '--task', args.task, '--outdir', str(outdir)], check=True)
print('[OK] predict_fz wrapper completed')
