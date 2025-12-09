import argparse, json, os
from pathlib import Path
import pandas as pd
import numpy as np
import subprocess
import pickle

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = REPO_ROOT / '.venv' / 'Scripts' / 'python.exe'

a = argparse.ArgumentParser(description='Predict 3D GRF (sandbox). Uses existing vGRF; AP/ML honest: NaN unless real model present.')
a.add_argument('--imu_csv', required=False)
a.add_argument('--outdir', required=True)
a.add_argument('--trial', required=True)
a.add_argument('--bw_kg', type=float, default=78.9)
a.add_argument('--window_ms', type=int, default=300)
a.add_argument('--model_pkl', default=None)
a.add_argument('--resume', action='store_true')
args = a.parse_args()

imu = Path(args.imu_csv) if args.imu_csv else None
outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
trial = args.trial

# Auto-discover model if not provided (prefer frozen HGB@300 kneepair)
def _auto_discover_model() -> str | None:
    rel = REPO_ROOT/'release'/'models'
    if rel.exists():
        for p in rel.rglob('*.pkl'):
            s = str(p).lower()
            if 'hgb' in s and 'w300' in s and 'kneepair' in s:
                return str(p)
    og = Path(r"E:\safestride\out_grid")
    if og.exists():
        for p in og.rglob('model.pkl'):
            s = str(p).lower()
            if 'ab01' in s and 'hgb' in s and 'w300' in s:
                return str(p)
    # Newly trained AP/ML model
    m = REPO_ROOT/'models'/'grf3d_hgb.pkl'
    if m.exists():
        return str(m)
    return None

model_path = args.model_pkl or _auto_discover_model()
if not model_path:
    raise SystemExit('No frozen model found for vGRF (need HGB@300); aborting')

# Step 1: ensure vGRF exists using existing pipeline (predict_fz.py)
pred_v_path = outdir / 'predicted_fz.csv'
if (not pred_v_path.exists() or not args.resume) and imu is not None:
    cmd = [str(PY), str(REPO_ROOT/'scripts'/'predict_fz.py'),
           '--imu_csv', str(imu), '--trial', trial, '--task', 'unknown', '--outdir', str(outdir)]
    if model_path:
        cmd += ['--model_pkl', str(model_path)]
    cmd += ['--bw_kg', str(args.bw_kg), '--window_ms', str(args.window_ms)]
    subprocess.run(cmd, check=False)

if not pred_v_path.exists():
    raise SystemExit(f'missing {pred_v_path}')

v = pd.read_csv(pred_v_path)
# Normalize columns
if 'Fz_%BW' not in v.columns:
    # derive from Fz_N and body mass
    if 'Fz_N' in v.columns and args.bw_kg:
        v['Fz_%BW'] = pd.to_numeric(v['Fz_N'], errors='coerce') / (args.bw_kg*9.80665) * 100.0
    else:
        v['Fz_%BW'] = np.nan

# Step 2: build 3D file with AP/ML honesty
n = len(v)
fx_bw = np.full(n, np.nan)
fy_bw = np.full(n, np.nan)
apml_status = 'not_available'

out = pd.DataFrame({
    'time_s': v.get('time_s', pd.Series(np.arange(n)/100.0)),
    'Fz_N': pd.to_numeric(v.get('Fz_N', pd.Series(np.nan)), errors='coerce'),
    'Fz_%BW': pd.to_numeric(v.get('Fz_%BW', pd.Series(np.nan)), errors='coerce'),
    'Fx_%BW': fx_bw,
    'Fy_%BW': fy_bw,
})

# If Fz_N is missing but %BW present, derive Newtons from %BW and body mass
try:
    if out['Fz_N'].isna().all() and 'Fz_%BW' in out.columns and np.isfinite(args.bw_kg):
        out['Fz_N'] = pd.to_numeric(out['Fz_%BW'], errors='coerce')/100.0 * (args.bw_kg * 9.80665)
except Exception:
    pass

# Predict AP/ML using trained model if available
z80 = 1.2816
rmse_v = 7.0
rmse_ap = 5.0
rmse_ml = 5.0
try:
    if args.imu_csv:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        imu_df = pd.read_csv(args.imu_csv)
        feat_cols = [c for c in imu_df.columns if str(c).startswith(('ax_','ay_','az_','gx_','gy_','gz_'))]
        if isinstance(model, dict) and all(k in model for k in ['scaler','model_ap','model_ml']):
            X = pd.DataFrame({c: pd.to_numeric(imu_df.get(c, pd.Series(np.nan, index=imu_df.index)), errors='coerce') for c in feat_cols}).fillna(0.0).values
            Xn = model['scaler'].transform(X)
            ap_hat = model['model_ap'].predict(Xn)
            ml_hat = model['model_ml'].predict(Xn)
            out['Fx_%BW'] = ap_hat
            out['Fy_%BW'] = ml_hat
            apml_status = 'available'
            rmse_ap = float(model.get('residual_std_ap', rmse_ap))
            rmse_ml = float(model.get('residual_std_ml', rmse_ml))
except Exception:
    # leave NaNs and not_available
    pass

# Bands
out['Fz_%BW_lo'] = out['Fz_%BW'] - z80*rmse_v
out['Fz_%BW_hi'] = out['Fz_%BW'] + z80*rmse_v
out['Fx_%BW_lo'] = out['Fx_%BW'] - z80*rmse_ap
out['Fx_%BW_hi'] = out['Fx_%BW'] + z80*rmse_ap
out['Fy_%BW_lo'] = out['Fy_%BW'] - z80*rmse_ml
out['Fy_%BW_hi'] = out['Fy_%BW'] + z80*rmse_ml

# Optional global calibration scaling
try:
    calib = json.loads((REPO_ROOT/'docs'/'vNext_multisignal'/'pi_calibration.json').read_text(encoding='utf-8'))
    for ax, col in [('vz','Fz_%BW'), ('ap','Fx_%BW'), ('ml','Fy_%BW')]:
        s = float(calib.get(ax, {}).get('scale', 1.0))
        if not np.isfinite(s) or s==1.0:
            continue
        lo = out[col+'_lo']; hi = out[col+'_hi']
        mid = (lo+hi)/2.0
        half = (hi - lo)/2.0
        half2 = half * s
        out[col+'_lo'] = mid - half2
        out[col+'_hi'] = mid + half2
except Exception:
    pass

out3 = outdir / 'predicted_fz3d.csv'
# also add Newton components derived from %BW if available
try:
    g = 9.80665
    if np.isfinite(args.bw_kg):
        out['Fx_N'] = pd.to_numeric(out['Fx_%BW'], errors='coerce')/100.0 * (args.bw_kg * g)
        out['Fy_N'] = pd.to_numeric(out['Fy_%BW'], errors='coerce')/100.0 * (args.bw_kg * g)
    else:
        out['Fx_N'] = np.nan
        out['Fy_N'] = np.nan
except Exception:
    out['Fx_N'] = np.nan
    out['Fy_N'] = np.nan
out['apml_status'] = apml_status
out.to_csv(out3, index=False)

# PI meta
meta = {
    'trial': trial,
    'window_ms': args.window_ms,
    'bw_kg': args.bw_kg,
    'pi': 0.8,
    'rmse_defaults': {'vz': rmse_v, 'ap': rmse_ap, 'ml': rmse_ml},
}
(outdir/'pi_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print('[OK] wrote', out3)
