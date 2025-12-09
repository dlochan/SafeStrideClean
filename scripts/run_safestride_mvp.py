import argparse, csv, os, re, subprocess, sys
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_CAND = REPO_ROOT / '.venv' / 'Scripts' / 'python.exe'
PY = PY_CAND if PY_CAND.exists() else Path(sys.executable)
# centralized roots (prefer E:)
try:
    from tools import path_config as PC  # type: ignore
except Exception:
    PC = None

ap = argparse.ArgumentParser(description='SafeStride MVP runner (resume-safe, frozen Clinical_v1 models only)')
ap.add_argument('--in_root', required=True, help='Root containing new IMU CSVs (*_imu*.csv)')
ap.add_argument('--out_root', default=None, help='Override output root; if omitted, use SafeStridePaths (env/cfg aware)')
ap.add_argument('--model_pkl', help='Path to frozen Clinical_v1 model (HGB@300). If omitted, auto-discover from release/models or out_grid')
ap.add_argument('--bw_kg_default', type=float, default=75.0)
ap.add_argument('--window_ms', type=int, default=300)
ap.add_argument('--force', action='store_true')
ap.add_argument('--resume', action='store_true')
ap.add_argument('--no_evidence_enforce', action='store_true')
ap.add_argument('--run_kin', action='store_true', help='Also build kinematic surrogates if IMU inputs available')
ap.add_argument('--run_risk', action='store_true', help='Also run risk engine v1 after metrics and surrogates are available')
ap.add_argument('--only_trials', help='Comma-separated list of trials or a file path containing one trial per line')
args = ap.parse_args()

paths = None
if PC is not None:
    try:
        paths = PC.SafeStridePaths.from_env_or_config()
    except Exception:
        paths = None

IN = Path(args.in_root)
if args.out_root:
    OUT = Path(args.out_root)
else:
    # Prefer SafeStridePaths (env/config aware) when available; fall back to prior default
    if paths is not None:
        OUT = paths.out_root
    else:
        OUT = REPO_ROOT/'mvp'/'out'
OUT.mkdir(parents=True, exist_ok=True)

if paths is not None:
    MVP_DIR = paths.doc_root/'mvp'
else:
    MVP_DIR = (PC.DOC_ROOT/'mvp') if PC is not None else (REPO_ROOT/'mvp')
MVP_DIR.mkdir(parents=True, exist_ok=True)
LEADER = MVP_DIR/'leaderboard_mvp.csv'
METRICS = MVP_DIR/'metrics_mvp.csv'
FLAGS = MVP_DIR/'flags_mvp.csv'

THRESH_PATH = REPO_ROOT/'configs'/'clinical_thresholds.yaml'

# dataset fs helper
def _get_fs_hz() -> float:
    try:
        import yaml
        cfg = REPO_ROOT/'configs'/'dataset.yaml'
        if cfg.exists():
            d = yaml.safe_load(cfg.read_text(encoding='utf-8')) or {}
            v = float(d.get('fs_hz', 200.0))
            if v and v > 0:
                return v
    except Exception:
        pass
    return 200.0

# 0) Evidence enforcement (unless disabled)
if not args.no_evidence_enforce:
    try:
        subprocess.run([str(PY), str(REPO_ROOT/'tools'/'evidence_registry.py'), '--enforce'], check=False)
    except Exception:
        pass

# 1) Model discovery

def _auto_discover_model() -> str | None:
    rel = (PC.RELEASE_ROOT/'models') if PC is not None else (REPO_ROOT/'release'/'models')
    if rel.exists():
        for p in rel.rglob('*.pkl'):
            s = str(p).lower()
            if 'hgb' in s and 'w300' in s and ('kneepair' in s or 'knee' in s or 'pair' in s):
                return str(p)
    og = Path(r"E:\safestride\out_grid")
    if og.exists():
        for p in og.rglob('model.pkl'):
            s = str(p).lower()
            if 'ab01' in s and 'hgb' in s and 'w300' in s:
                return str(p)
    # fallback to trained baseline
    m = REPO_ROOT/'models'/'grf_baseline_hgb_w300.pkl'
    if m.exists():
        return str(m)
    return None

MODEL = args.model_pkl or _auto_discover_model()
if not MODEL:
    raise SystemExit('No frozen model found (HGB@300); aborting')

# 2) Scan for IMU files

def iter_imu_files(root: Path):
    pats = ['*_imu_real.csv', '*_imu.csv', '*_imu_real.csv.gz', '*_imu.csv.gz', '*.csv', '*.csv.gz']
    seen = set()
    for pat in pats:
        for p in root.rglob(pat):
            sp = str(p)
            if sp in seen:
                continue
            seen.add(sp)
            if p.name.lower().endswith('_grf.csv') or p.name.lower().endswith('_grf.csv.gz'):
                continue
            yield p

def has_time_s_column(p: Path) -> bool:
    try:
        # read only header/first row
        df = pd.read_csv(p, nrows=1)
        return 'time_s' in df.columns
    except Exception:
        return False

def has_valid_imu_axes(p: Path) -> bool:
    try:
        import sys as _sys
        if str(REPO_ROOT) not in _sys.path:
            _sys.path.append(str(REPO_ROOT))
        from src.imu_schema import normalize_imu_columns, ensure_time_s
        # load a small chunk or full file safely
        df = pd.read_csv(p)
        df = normalize_imu_columns(df)
        fs = _get_fs_hz()
        df = ensure_time_s(df, fs)
        # eligibility: at least accelerometer axes present (gyros optional)
        base_acc = ['ax','ay','az']
        single_ok = all(c in df.columns for c in base_acc)
        if single_ok:
            return True
        # detect any complete 3-axis accelerometer set with suffix
        tags = {}
        for c in df.columns:
            for k in base_acc:
                pref = f"{k}_"
                if str(c).startswith(pref):
                    tag = str(c)[len(pref):]
                    tags.setdefault(tag, set()).add(k)
        return any(len(s)==3 for s in tags.values())
    except Exception as e:
        # log reason via shared log
        try:
            from tools import path_config as _PC
            with open((_PC.LOG_ROOT/'validation_mvp_risk.log'), 'a', encoding='utf-8') as f:
                f.write(f"[MVP] ineligible {p.name}: {e}\n")
        except Exception:
            pass
        return False

def trial_from_name(p: Path) -> str | None:
    name = p.name
    for suf in ['_imu_real.csv', '_imu.csv', '_imu_real.csv.gz', '_imu.csv.gz']:
        if name.endswith(suf):
            return name[: -len(suf)]
    if name.endswith('.csv'):
        return name[:-4]
    if name.endswith('.csv.gz'):
        return name[:-7]
    return None

def subject_from_trial(trial: str) -> str:
    m = re.search(r"P\d{2}_S\d{2}", trial, flags=re.IGNORECASE)
    if m:
        return m.group(0)[:7]
    m2 = re.search(r"P\d{2}", trial, flags=re.IGNORECASE)
    if m2:
        return m2.group(0)[:3]
    if len(trial) >= 4 and re.match(r"[A-Za-z]{2}\d{2}", trial):
        return trial[:4]
    return trial[:8]

def side_from_name(trial: str) -> str | None:
    t = trial.lower()
    if any(k in t for k in ['_left', '-left', ' left']):
        return 'left'
    if any(k in t for k in ['_right', '-right', ' right']):
        return 'right'
    if any(k in t for k in ['_l_', '_l-']) or t.endswith('_l'):
        return 'left'
    if any(k in t for k in ['_r_', '_r-']) or t.endswith('_r'):
        return 'right'
    return None

def session_date_for(p: Path) -> str:
    try:
        ts = p.stat().st_mtime
        from datetime import datetime
        return datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    except Exception:
        return '1970-01-01'

# Normalizer loader (no package assumption)
def _normalize_imu(imu_csv: Path, out_csv: Path | None) -> tuple[Path, dict]:
    import importlib.util
    mod_path = REPO_ROOT / 'tools' / 'normalize_imu_schema.py'
    spec = importlib.util.spec_from_file_location('normalize_imu_schema', str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot_load_normalizer')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    df, rec = mod.normalize_file(Path(imu_csv))  # type: ignore
    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return (out_csv if out_csv else imu_csv), rec  # path to use

# threshold loader
try:
    import yaml
    TH = yaml.safe_load(THRESH_PATH.read_text(encoding='utf-8')) or {}
    MVP_TH = TH.get('mvp', {})
except Exception:
    MVP_TH = {}

rows_leader = []
rows_metrics = []
rows_flags = []

# optional trial allowlist
ALLOW: set[str] | None = None
if args.only_trials:
    s = args.only_trials
    ALLOW = set()
    p = Path(s)
    try:
        if p.exists():
            for line in p.read_text(encoding='utf-8').splitlines():
                t = line.strip()
                if t:
                    ALLOW.add(t)
        else:
            for t in s.split(','):
                t2 = t.strip()
                if t2:
                    ALLOW.add(t2)
    except Exception:
        # if parsing fails, ignore filter
        ALLOW = None

# helper metrics

def compute_time_vec(df: pd.DataFrame) -> np.ndarray:
    if 'time_s' in df.columns:
        t = pd.to_numeric(df['time_s'], errors='coerce').to_numpy()
        if np.isfinite(t).sum() >= 2:
            return t
    # fallback to dataset fs
    dt = 1.0/max(_get_fs_hz(), 1.0)
    n = len(df)
    return np.arange(n) * dt

def compute_metrics_for_trial(pred_csv: Path, bw_kg: float) -> dict:
    d = pd.read_csv(pred_csv)
    t = compute_time_vec(d)
    dt = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 1/200.0
    y = None
    if 'Fz_%BW' in d.columns:
        y = pd.to_numeric(d['Fz_%BW'], errors='coerce').to_numpy()
    elif 'Fz_N' in d.columns and bw_kg:
        y = pd.to_numeric(d['Fz_N'], errors='coerce').to_numpy() / (bw_kg*9.80665) * 100.0
    else:
        y = np.full(len(d), np.nan)
    # peak %BW
    peak = float(np.nanmax(y)) if len(y) else float('nan')
    # stance mask > 5% BW
    m = np.isfinite(y) & (y > 5.0)
    # stance time (longest contiguous block)
    stance_time = 0.0
    if m.any():
        idx = np.where(m)[0]
        # contiguous runs
        run_len = 1; best = 1
        for i in range(1, len(idx)):
            if idx[i] == idx[i-1] + 1:
                run_len += 1
                best = max(best, run_len)
            else:
                run_len = 1
        stance_time = best * dt
    # impulse (%BW*s)
    impulse = float(np.nansum(np.where(m, y, 0.0))) * dt
    # loading rate: max positive diff
    if len(y) > 1:
        dy = np.diff(y) / dt
        loading_rate_pctbw_s = float(np.nanmax(dy))
    else:
        loading_rate_pctbw_s = float('nan')
    loading_rate_Ns = float('nan')
    if np.isfinite(loading_rate_pctbw_s) and bw_kg:
        loading_rate_Ns = loading_rate_pctbw_s/100.0 * (bw_kg*9.80665)
    return {
        'peak_pctbw': peak,
        'stance_time_s': float(stance_time),
        'impulse_pctbw_s': float(impulse),
        'loading_rate_pctbw_s': float(loading_rate_pctbw_s),
        'loading_rate_Ns': float(loading_rate_Ns),
    }

# process trials
by_subject_day = {}
by_subject_task_side = {}

for imu_csv in iter_imu_files(IN):
    trial = trial_from_name(imu_csv)
    if not trial:
        continue
    if ALLOW is not None and trial not in ALLOW:
        continue
    # Normalize schema deterministically; exclude with reason on failure
    normalized_csv = None
    try:
        outdir = OUT / trial
        outdir.mkdir(parents=True, exist_ok=True)
        normalized_csv = outdir / 'imu_normalized.csv'
        normalized_csv, rec = _normalize_imu(imu_csv, normalized_csv)
    except Exception as e:
        # record reason
        reason = str(e)
        try:
            from tools import path_config as _PC
            with open((_PC.LOG_ROOT/'validation_mvp_risk.log'), 'a', encoding='utf-8') as f:
                f.write(f"[MVP] ineligible {imu_csv.name}: {reason}\n")
        except Exception:
            pass
        continue
    subj = subject_from_trial(trial)
    side = side_from_name(trial)
    day = session_date_for(imu_csv)
    outdir = OUT / trial
    outdir.mkdir(parents=True, exist_ok=True)
    pred_csv = outdir/'predicted_fz.csv'
    if args.resume and (not args.force) and pred_csv.exists():
        pass
    else:
        # predict (frozen Clinical_v1 model)
        cmd = [str(PY), str(REPO_ROOT/'scripts'/'predict_fz.py'),
               '--imu_csv', str(normalized_csv), '--model_pkl', str(MODEL),
               '--bw_kg', str(args.bw_kg_default), '--window_ms', str(args.window_ms),
               '--trial', trial, '--task', 'unknown', '--outdir', str(outdir)]
        rc = subprocess.run(cmd, check=False)
        if rc.returncode != 0 and not pred_csv.exists():
            continue
    # metrics
    if pred_csv.exists():
        met = compute_metrics_for_trial(pred_csv, args.bw_kg_default)
        rows_metrics.append({'trial': trial, 'subject': subj, 'session_date': day, 'side': side, **met})
        rows_leader.append({'trial': trial, 'outdir': str(outdir), 'bw_kg': args.bw_kg_default, 'window_ms': args.window_ms})
        # accumulators for asymmetry
        if side in ('left','right'):
            key = (subj, day, 'unknown')
            rec = by_subject_task_side.get(key, {'left': None, 'right': None})
            rec[side] = met
            by_subject_task_side[key] = rec
        # accumulators for longitudinal
        keyd = (subj, day)
        agg = by_subject_day.get(keyd, {'impulse': 0.0, 'peak': 0.0})
        agg['impulse'] += float(met['impulse_pctbw_s']) if math.isfinite(met['impulse_pctbw_s']) else 0.0
        agg['peak'] = max(float(met['peak_pctbw']), agg['peak'])
        by_subject_day[keyd] = agg

# asymmetry flags
ai_rows = []
for (subj, day, task), rec in by_subject_task_side.items():
    L = rec.get('left'); R = rec.get('right')
    if not L or not R:
        continue
    # AI on peak
    pL = L['peak_pctbw']; pR = R['peak_pctbw']
    if all(math.isfinite(v) for v in [pL,pR]) and max(pL,pR)>0:
        ai = abs(pL - pR) / max(pL, pR)
        ai_rows.append({'subject': subj, 'session_date': day, 'ai_peak': float(ai)})
        th = MVP_TH.get('elevated_asymmetry', {})
        thr = float(th.get('ai_max', 0.2))
        if ai > thr:
            rows_flags.append({'trial': f"{subj}_{day}", 'subject': subj, 'flag': 'Elevated asymmetry', 'value': float(ai), 'threshold': thr, 'direction': 'high', 'source': th.get('source',''), 'evidence_grade': th.get('evidence_grade','')})

# longitudinal AC ratio flags
from datetime import datetime, timedelta

for subj in sorted(set(s for (s,_) in by_subject_day.keys())):
    # build series by date
    days = sorted(d for (s,d) in by_subject_day.keys() if s==subj)
    if not days:
        continue
    def to_dt(x):
        try: return datetime.strptime(x, '%Y-%m-%d')
        except: return None
    day_dt = [to_dt(d) for d in days if to_dt(d)]
    if not day_dt:
        continue
    min_d, max_d = min(day_dt), max(day_dt)
    # evaluate only on last day for flagging
    last = max_d
    def sum_window(end_dt, days_back, key):
        s = 0.0
        start = end_dt - timedelta(days=days_back)
        for (s0,d0), agg in by_subject_day.items():
            if s0!=subj: continue
            dt0 = to_dt(d0)
            if not dt0: continue
            if start < dt0 <= end_dt:
                s += float(agg[key])
        return s
    ac_thr = MVP_TH.get('chronic_load_high', {})
    ac_max = float(ac_thr.get('ac_ratio_max', 1.5))
    acute = sum_window(last, 7, 'impulse')
    chronic = sum_window(last, 28, 'impulse')/4.0 if sum_window(last, 28, 'impulse')>0 else 0.0
    ac = float('inf') if chronic==0 else acute/chronic
    if math.isfinite(ac) and ac > ac_max:
        rows_flags.append({'trial': f"{subj}_{last.strftime('%Y-%m-%d')}", 'subject': subj, 'flag': 'Chronic load high', 'value': float(ac), 'threshold': ac_max, 'direction': 'high', 'source': ac_thr.get('source',''), 'evidence_grade': ac_thr.get('evidence_grade','')})

# per-trial flags: high impact & stiff landing
for r in rows_metrics:
    th_hi = MVP_TH.get('high_impact_loading', {})
    peak_thr = float(th_hi.get('peak_vgrf_pctbw_max', 400.0))
    if math.isfinite(r['peak_pctbw']) and r['peak_pctbw'] > peak_thr:
        rows_flags.append({'trial': r['trial'], 'subject': r['subject'], 'flag': 'High impact loading', 'value': float(r['peak_pctbw']), 'threshold': peak_thr, 'direction': 'high', 'source': th_hi.get('source',''), 'evidence_grade': th_hi.get('evidence_grade','')})
    th_sl = MVP_TH.get('stiff_landing', {})
    # prefer N/s when BW known
    lrN_thr = float(th_sl.get('loading_rate_Ns_max', th_sl.get('loading_rate_Ns', 20000))) if 'loading_rate_Ns' in th_sl or 'loading_rate_Ns_max' in th_sl else None
    lrBW_thr = float(th_sl.get('loading_rate_pctbw_s_max', 3000.0))
    ttp_ms_max = float(th_sl.get('ttp_ms_max', 60))
    # estimate time-to-peak crudely from samples
    ttp = float('nan')
    # could compute if we still have access to waveform here; skip due to IO cost
    # use loading-rate only criterion here
    if lrN_thr and math.isfinite(r['loading_rate_Ns']) and r['loading_rate_Ns'] > lrN_thr:
        rows_flags.append({'trial': r['trial'], 'subject': r['subject'], 'flag': 'Stiff landing', 'value': float(r['loading_rate_Ns']), 'threshold': lrN_thr, 'direction': 'high', 'source': th_sl.get('source',''), 'evidence_grade': th_sl.get('evidence_grade','')})
    elif math.isfinite(r['loading_rate_pctbw_s']) and r['loading_rate_pctbw_s'] > lrBW_thr:
        rows_flags.append({'trial': r['trial'], 'subject': r['subject'], 'flag': 'Stiff landing', 'value': float(r['loading_rate_pctbw_s']), 'threshold': lrBW_thr, 'direction': 'high', 'source': th_sl.get('source',''), 'evidence_grade': th_sl.get('evidence_grade','')})

# write outputs (idempotent)
pd.DataFrame(rows_leader).to_csv(LEADER, index=False)
pd.DataFrame(rows_metrics).to_csv(METRICS, index=False)
pd.DataFrame(rows_flags).to_csv(FLAGS, index=False)
# write meta
try:
    meta = {'model_pkl': str(MODEL), 'n_trials': len(rows_leader)}
    meta_path = MVP_DIR/'mvp_meta.json'
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')
except Exception:
    pass
print('[OK] MVP wrote', LEADER, METRICS, FLAGS)

# Optional: Kinematics Engine v1
if args.run_kin:
    try:
        subprocess.run([str(PY), str(REPO_ROOT/'tools'/'build_kinematic_surrogates.py'),
                        '--in_root', str(IN), '--pred_root', str(OUT), '--resume'], check=False)
    except Exception:
        pass

# Optional: Risk Engine v1
if args.run_risk:
    try:
        subprocess.run([str(PY), str(REPO_ROOT/'tools'/'risk_engine_v1.py'), '--resume'], check=False)
    except Exception:
        pass
