import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import pickle

REPO_ROOT = Path(r"C:\Users\locha\Documents\safestride")
MODELS = REPO_ROOT / 'models'
MODELS.mkdir(parents=True, exist_ok=True)

ap = argparse.ArgumentParser(description='Train 3D GRF models (vz via baseline; train AP/ML from truth if available)')
ap.add_argument('--in_root', required=True)
ap.add_argument('--work_root', default=str(REPO_ROOT/'data'/'working'))
ap.add_argument('--outfile', default=str(MODELS/'grf3d_hgb.pkl'))
ap.add_argument('--resume', action='store_true')
args = ap.parse_args()

out = Path(args.outfile)
if args.resume and out.exists():
    print('[OK] grf3d model exists', out)
    raise SystemExit(0)

in_root = Path(args.in_root)
work_root = Path(args.work_root)

def iter_pairs():
    # Yield (trial, imu_csv, truth3d_csv)
    # Prefer adapted IMU under out_grid_multisignal/grf3d/<trial>/adapted_imu.csv if exists
    # else *_imu_real.csv under in_root
    for imu in in_root.rglob('*_imu_real.csv'):
        trial = imu.stem.replace('_imu_real','')
        # truth priority: in work_root, else alongside imu as *_grf3d_truth.csv
        t1 = work_root / f"{trial}_grf3d_truth.csv"
        t2 = imu.with_name(trial + '_grf3d_truth.csv')
        truth = t1 if t1.exists() else (t2 if t2.exists() else None)
        if truth is None:
            continue
        # prefer adapted imu produced by run_grf3d
        alt = REPO_ROOT/'out_grid_multisignal'/'grf3d'/trial/'adapted_imu.csv'
        imu_use = alt if alt.exists() else imu
        yield trial, imu_use, truth

def build_features(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if c.startswith(('ax_','ay_','az_','gx_','gy_','gz_'))]
    X = pd.DataFrame({c: pd.to_numeric(df[c], errors='coerce') for c in cols}).fillna(0.0).values
    return X, cols

trials = []
X_list = []
y_ap = []
y_ml = []
BW_N = 78.9*9.80665
for trial, imu_csv, truth_csv in iter_pairs():
    try:
        df_imu = pd.read_csv(imu_csv)
        df_tr = pd.read_csv(truth_csv)
    except Exception:
        continue
    n = min(len(df_imu), len(df_tr))
    if n < 100:
        continue
    df_imu = df_imu.iloc[:n]
    df_tr = df_tr.iloc[:n]
    X, feat_cols = build_features(df_imu)
    X_list.append(X)
    # targets in %BW for AP/ML if present
    fx = pd.to_numeric(df_tr.get('Fx_N', pd.Series(np.nan, index=range(n))), errors='coerce').fillna(0.0).values
    fy = pd.to_numeric(df_tr.get('Fy_N', pd.Series(np.nan, index=range(n))), errors='coerce').fillna(0.0).values
    y_ap.append((fx/BW_N)*100.0)
    y_ml.append((fy/BW_N)*100.0)
    trials.append(trial)

if not X_list:
    # No truth found; write stub meta to keep pipeline moving
    out.write_text(json.dumps({'kind':'stub','note':'no truth found; AP/ML fall back to zeros'}, indent=2), encoding='utf-8')
    print('[WARN] no truth found; wrote stub meta', out)
    raise SystemExit(0)

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
except Exception:
    out.write_text(json.dumps({'kind':'stub','note':'sklearn unavailable; AP/ML fall back to zeros'}, indent=2), encoding='utf-8')
    print('[WARN] sklearn unavailable; wrote stub meta', out)
    raise SystemExit(0)

X_all = np.vstack(X_list)
y_ap_all = np.concatenate(y_ap)
y_ml_all = np.concatenate(y_ml)
scaler = StandardScaler()
Xn = scaler.fit_transform(X_all)

reg_ap = Ridge(alpha=1.0, random_state=0)
reg_ml = Ridge(alpha=1.0, random_state=0)
reg_ap.fit(Xn, y_ap_all)
reg_ml.fit(Xn, y_ml_all)

# residual std for PI bands
pred_ap = reg_ap.predict(Xn)
pred_ml = reg_ml.predict(Xn)
res_ap = float(np.std(y_ap_all - pred_ap))
res_ml = float(np.std(y_ml_all - pred_ml))

model = {
    'features': feat_cols,
    'scaler': scaler,
    'model_ap': reg_ap,
    'model_ml': reg_ml,
    'residual_std_ap': res_ap,
    'residual_std_ml': res_ml,
}
with open(out, 'wb') as f:
    pickle.dump(model, f)
print('[OK] trained GRF3D (AP/ML) on frames=', X_all.shape[0], 'outfile=', out)
