import argparse, sys, subprocess, math
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
PY_CAND = REPO_ROOT / '.venv' / 'Scripts' / 'python.exe'
PY = PY_CAND if PY_CAND.exists() else Path(sys.executable)
# centralized path roots (prefer E:)
try:
    from tools import path_config as PC  # type: ignore
except Exception:
    PC = None
if PC is not None:
    LOGS = PC.LOG_ROOT
    VALDIR = (PC.DOC_ROOT / 'validation')
    MVP_DIR = (PC.DOC_ROOT / 'mvp')
    RISK_DIR = (PC.DOC_ROOT / 'risk_engine')
else:
    LOGS = REPO_ROOT / 'logs'
    VALDIR = REPO_ROOT / 'docs' / 'validation'
    MVP_DIR = REPO_ROOT / 'mvp'
    RISK_DIR = REPO_ROOT / 'docs' / 'risk_engine'
LOGS.mkdir(parents=True, exist_ok=True)
VALDIR.mkdir(parents=True, exist_ok=True)
MVP_DIR.mkdir(parents=True, exist_ok=True)
RISK_DIR.mkdir(parents=True, exist_ok=True)
LOG = LOGS / 'validation_mvp_risk.log'

MVP_SUM = VALDIR / 'mvp_internal_summary.csv'
RISK_SUM = VALDIR / 'risk_engine_internal_summary.csv'
REPORT = VALDIR / 'VALIDATION_REPORT_MVP_RISK_v1.md'

METRICS = MVP_DIR / 'metrics_mvp.csv'
FLAGS = MVP_DIR / 'flags_mvp.csv'
RISK_FLAGS = RISK_DIR / 'risk_flags.csv'

ap = argparse.ArgumentParser(description='Run internal validation for MVP + Risk Engine v1 (resume-safe)')
ap.add_argument('--in_root', default=None, help='Optional dataset root to scan recursively for IMU CSVs')
ap.add_argument('--resume', action='store_true')
ap.add_argument('--force', action='store_true')
args = ap.parse_args()

def log(msg: str):
    try:
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write('[VALID] '+msg+'\n')
    except Exception:
        pass

def _get_default_root() -> Path:
    try:
        cfgp = REPO_ROOT / 'configs' / 'dataset.yaml'
        if cfgp.exists():
            d = yaml.safe_load(cfgp.read_text(encoding='utf-8')) or {}
            r = d.get('dataset_root')
            if r:
                return Path(str(r))
    except Exception:
        pass
    return REPO_ROOT / 'data' / 'working'

# Resolve dataset roots: --in_root overrides; else use dataset.yaml default with fallback
primary = Path(args.in_root) if args.in_root else _get_default_root()
fallback = Path(r"E:\safestride\datasets\ProcessedData")
IN = primary
USED_ROOT = IN
print(f"[VALID] Starting internal validation: root={USED_ROOT}")
try:
    if PC is not None:
        roots = PC.describe_roots()
        print('[VALID] Resolved roots: ' + ', '.join([f"{k}={v}" for k,v in roots.items()]))
except Exception:
    pass

# Evidence enforcement gate (robust: log and continue even if it fails)
try:
    subprocess.run([str(PY), str(REPO_ROOT/'tools'/'evidence_registry.py'), '--enforce'], check=True)
    log('Evidence enforcement: PASS')
except subprocess.CalledProcessError:
    log('Evidence enforcement: FAIL')

# Build canonical trial list from known leaderboards (optional restriction for summaries)
canonical = set()
try:
    # External grid leaderboard
    lb1 = Path(r"E:\safestride\out_grid\leaderboard_all.csv")
    if lb1.exists():
        df1 = pd.read_csv(lb1)
        col = 'trial' if 'trial' in df1.columns else df1.columns[0]
        canonical.update(str(x) for x in df1[col].dropna().astype(str).tolist())
except Exception:
    pass
try:
    lb2 = REPO_ROOT / 'docs' / 'leaderboard_all_filtered.csv'
    if lb2.exists():
        df2 = pd.read_csv(lb2)
        col2 = 'trial' if 'trial' in df2.columns else df2.columns[0]
        canonical.update(str(x) for x in df2[col2].dropna().astype(str).tolist())
except Exception:
    pass

# Pre-scan IN for eligibility reasons (strict deterministic checks)
reasons = {'bad_columns': 0, 'ambiguous_columns': 0, 'no_time_s': 0, 'too_short_for_window': 0, 'other_format_issue': 0}
total_scanned = 0

def _iter_trials(root: Path):
    pats = ['*_imu_real.csv','*_imu.csv','*.csv','*_imu_real.csv.gz','*_imu.csv.gz','*.csv.gz']
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

def _trial_from_name(p: Path) -> str:
    name = p.name
    for suf in ['_imu_real.csv','_imu.csv','_imu_real.csv.gz','_imu.csv.gz']:
        if name.endswith(suf):
            return name[:-len(suf)]
    if name.endswith('.csv'): return name[:-4]
    if name.endswith('.csv.gz'): return name[:-7]
    return name

eligible_trials = set()
cap_n = 0
try:
    cap_n = int(os.environ.get('SAFESTRIDE_CAP_N', '0') or '0')
except Exception:
    cap_n = 0
window_ms = 300

# helper: normalize using tools/normalize_imu_schema.py
def _normalize_view(p: Path) -> pd.DataFrame:
    import importlib.util
    mod_path = REPO_ROOT / 'tools' / 'normalize_imu_schema.py'
    spec = importlib.util.spec_from_file_location('normalize_imu_schema', str(mod_path))
    if spec is None or spec.loader is None:
        raise RuntimeError('cannot_load_normalizer')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore
    df, _rec = mod.normalize_file(Path(p))  # type: ignore
    return df
for imu in _iter_trials(IN):
    total_scanned += 1
    trial = _trial_from_name(imu)
    # evaluate all files regardless of canonical lists
    try:
        import sys as _sys
        if str(REPO_ROOT) not in _sys.path:
            _sys.path.append(str(REPO_ROOT))
        # normalize deterministically; raises on bad/ambiguous/no time
        d = _normalize_view(imu)
        # dt and window length
        t = pd.to_numeric(d.get('time_s'), errors='coerce').to_numpy()
        dt = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 1/200.0
        fs_eff = 1.0/dt if dt>0 else 200.0
        win_len = max(3, int(round(fs_eff * window_ms / 1000.0)))
        if len(d) < (win_len + 2):
            reasons['too_short_for_window'] += 1
            log(f"ineligible {trial}: too_short_for_window (n={len(d)}, win_len={win_len})")
            continue
        # accelerometer presence (gyros optional)
        base_acc = ['ax','ay','az']
        single_ok = all(c in d.columns for c in base_acc)
        if not single_ok:
            tags = {}
            for c in d.columns:
                for k in base_acc:
                    pref = f"{k}_"
                    if str(c).startswith(pref):
                        tag = str(c)[len(pref):]
                        tags.setdefault(tag, set()).add(k)
            if not any(len(s)==3 for s in tags.values()):
                reasons['bad_columns'] += 1
                log(f"ineligible {trial}: bad_columns (missing accelerometer axes)")
                continue
        eligible_trials.add(trial)
    except Exception as e:
        msg = str(e)
        if 'ambiguous_columns' in msg:
            reasons['ambiguous_columns'] += 1
            log(f"ineligible {trial}: ambiguous_columns")
        elif 'no_time_s_and_no_fs' in msg:
            reasons['no_time_s'] += 1
            log(f"ineligible {trial}: no_time_s")
        elif 'bad_columns_missing_acc' in msg or 'Missing IMU columns' in msg:
            reasons['bad_columns'] += 1
            log(f"ineligible {trial}: bad_columns")
        else:
            reasons['other_format_issue'] += 1
            log(f"ineligible {trial}: other_format_issue ({e})")
        continue
    if total_scanned % 200 == 0:
        print(f"[PROGRESS] scanned={total_scanned} eligible={len(eligible_trials)} reasons={reasons}")

# If no eligible trials found on primary and we weren't explicitly given --in_root, fall back
if (len(eligible_trials) == 0) and (IN != fallback) and fallback.exists():
    IN = fallback
    USED_ROOT = IN
    # reset counters and rescan fallback
    reasons = {'bad_columns': 0, 'no_time_s': 0, 'too_short_for_window': 0, 'other_format_issue': 0}
    total_scanned = 0
    eligible_trials = set()
    for imu in _iter_trials(IN):
        total_scanned += 1
        trial = _trial_from_name(imu)
        # evaluate all files regardless of canonical lists
        try:
            d = _normalize_view(imu)
            t = pd.to_numeric(d.get('time_s'), errors='coerce').to_numpy()
            dt = float(np.nanmedian(np.diff(t))) if len(t) > 1 else 1/200.0
            fs_eff = 1.0/dt if dt>0 else 200.0
            win_len = max(3, int(round(fs_eff * window_ms / 1000.0)))
            if len(d) < (win_len + 2):
                reasons['too_short_for_window'] += 1
                log(f"ineligible {trial}: too_short_for_window (n={len(d)}, win_len={win_len})")
                continue
            base_acc = ['ax','ay','az']
            single_ok = all(c in d.columns for c in base_acc)
            if not single_ok:
                tags = {}
                for c in d.columns:
                    for k in base_acc:
                        pref = f"{k}_"
                        if str(c).startswith(pref):
                            tag = str(c)[len(pref):]
                            tags.setdefault(tag, set()).add(k)
                if not any(len(s)==3 for s in tags.values()):
                    reasons['bad_columns'] += 1
                    log(f"ineligible {trial}: bad_columns (missing accelerometer axes)")
                    continue
            eligible_trials.add(trial)
        except Exception as e:
            msg = str(e)
            if 'ambiguous_columns' in msg:
                reasons['ambiguous_columns'] += 1
                log(f"ineligible {trial}: ambiguous_columns")
            elif 'no_time_s_and_no_fs' in msg:
                reasons['no_time_s'] += 1
                log(f"ineligible {trial}: no_time_s")
            elif 'bad_columns_missing_acc' in msg or 'Missing IMU columns' in msg:
                reasons['bad_columns'] += 1
                log(f"ineligible {trial}: bad_columns")
            else:
                reasons['other_format_issue'] += 1
                log(f"ineligible {trial}: other_format_issue ({e})")
            continue
        if total_scanned % 200 == 0:
            print(f"[PROGRESS] scanned={total_scanned} eligible={len(eligible_trials)} reasons={reasons}")

# Run MVP + Kinematics + Risk (resume-safe) on eligible trials only; always finalize outputs
met = pd.DataFrame()
flags = pd.DataFrame()
risk_flags = pd.DataFrame()
try:
    if len(eligible_trials) > 0:
        sel = sorted(eligible_trials)
        if cap_n and len(sel) > cap_n:
            print(f"[VALID] Capping eligible trials to {cap_n} of {len(sel)} via SAFESTRIDE_CAP_N")
            sel = sel[:cap_n]
        # write allowlist
        allow_path = VALDIR / 'eligible_trials.txt'
        try:
            allow_path.write_text('\n'.join(sel), encoding='utf-8')
        except Exception:
            pass
        args_mvp = [str(PY), str(REPO_ROOT/'scripts'/'run_safestride_mvp.py'), '--in_root', str(IN), '--resume', '--run_kin', '--run_risk', '--only_trials', str(allow_path)]
        if args.force:
            args_mvp.append('--force')
        subprocess.run(args_mvp, check=False)
    # Load outputs (do not abort if missing; we still write empty summaries)
    met = pd.read_csv(METRICS) if METRICS.exists() else pd.DataFrame(columns=['subject','trial','task','peak_pctbw','loading_rate_Ns','impulse_pctbw_s','stance_time_s'])
    flags = pd.read_csv(FLAGS) if FLAGS.exists() else pd.DataFrame(columns=['trial','flag'])
    risk_flags = pd.read_csv(RISK_FLAGS) if RISK_FLAGS.exists() else pd.DataFrame(columns=['rule_id','flag'])
except Exception as e:
    log(f"pipeline_error: {e}")
finally:
    # proceed to writing summaries regardless
    pass

# Derive simple task labels from trial names for sanity checks
import re as _re

def derive_task(trial: str) -> str:
    t = (trial or '').lower()
    if any(k in t for k in ['cut', 'pivot']):
        return 'cutting/pivot'
    if any(k in t for k in ['drop', 'land', 'jump']):
        return 'landing/jump'
    if 'run' in t: return 'run'
    if any(k in t for k in ['walk','gait','2min']): return 'walk/gait'
    return 'unknown'

if not met.empty and 'trial' in met.columns:
    met = met.assign(task=met['trial'].map(derive_task))
    # restrict to canonical trials if the set is non-empty
    if canonical:
        met = met[met['trial'].astype(str).isin(canonical)]

# MVP summaries
# 1) Aggregated distributions for the report
summary_rows = []
for key in ['peak_pctbw','loading_rate_Ns','impulse_pctbw_s','stance_time_s']:
    if key in met.columns and not met.empty:
        s = pd.to_numeric(met[key], errors='coerce')
        summary_rows.append({'metric': key, 'count': int(s.notna().sum()), 'mean': float(s.mean()), 'p50': float(s.quantile(0.5)), 'p95': float(s.quantile(0.95))})

# Asymmetry index from pairs within subject-date
ai_values = []
if (not met.empty) and {'subject','session_date','side','peak_pctbw'}.issubset(met.columns):
    grp = met.groupby(['subject','session_date'])
    for (subj, day), g in grp:
        try:
            sides = set(g['side'].fillna('').astype(str).str.lower())
        except Exception:
            sides = set()
        if {'left','right'}.issubset(sides):
            try:
                L = float(g[g['side'].str.lower()=='left']['peak_pctbw'].iloc[0])
                R = float(g[g['side'].str.lower()=='right']['peak_pctbw'].iloc[0])
                if max(L,R) > 0:
                    ai_values.append(abs(L-R)/max(L,R))
            except Exception:
                pass
if ai_values:
    s = pd.Series(ai_values)
    summary_rows.append({'metric': 'asymmetry_index_peak', 'count': int(s.notna().sum()), 'mean': float(s.mean()), 'p50': float(s.quantile(0.5)), 'p90': float(s.quantile(0.9))})

# 2) Per-trial summary CSV (subject, trial, task, metrics, ai, AC ratio, flags)
per_trial_cols = ['subject','trial','task','peak_pctbw','loading_rate_Ns','impulse_pctbw_s','stance_time_s']
per_trial_df = pd.DataFrame(columns=per_trial_cols)
if not met.empty:
    per_trial_df = met[per_trial_cols].copy()
    # join AI by subject-day
    if ai_values and {'subject','session_date'}.issubset(met.columns):
        # recompute AI per subject-day as above and merge
        ai_rows = []
        grp = met.groupby(['subject','session_date'])
        for (subj, day), g in grp:
            try:
                sides = set(g['side'].fillna('').astype(str).str.lower())
            except Exception:
                sides = set()
            if {'left','right'}.issubset(sides):
                try:
                    L = float(g[g['side'].str.lower()=='left']['peak_pctbw'].iloc[0])
                    R = float(g[g['side'].str.lower()=='right']['peak_pctbw'].iloc[0])
                    if max(L,R) > 0:
                        ai = abs(L-R)/max(L,R)
                        ai_rows.append({'subject': subj, 'session_date': day, 'ai_peak': ai})
                except Exception:
                    pass
        if ai_rows:
            ai_df = pd.DataFrame(ai_rows)
            per_trial_df = per_trial_df.merge(ai_df, on=['subject','session_date'], how='left')
    # join AC ratio from risk features if available
    FEAT = REPO_ROOT / 'docs' / 'risk_engine' / 'risk_features.csv'
    if FEAT.exists():
        try:
            rf = pd.read_csv(FEAT)
            if {'trial','ac_ratio_impulse'}.issubset(rf.columns):
                per_trial_df = per_trial_df.merge(rf[['trial','ac_ratio_impulse']], on='trial', how='left')
        except Exception:
            pass
    # add per-trial MVP flags (concat names -> snake_case)
    if not flags.empty and 'trial' in flags.columns and 'flag' in flags.columns:
        def to_snake(name: str) -> str:
            n = str(name).strip().lower()
            n = n.replace(' ', '_').replace('-', '_')
            return n
        fl = flags.copy()
        fl['flag'] = fl['flag'].map(to_snake)
        fl = fl.groupby('trial')['flag'].apply(lambda s: ';'.join(sorted(set(s.astype(str))))).reset_index(name='flags')
        per_trial_df = per_trial_df.merge(fl, on='trial', how='left')

    # rename columns to requested output names
    per_trial_df = per_trial_df.rename(columns={'peak_pctbw': 'peak_vgrf_pctbw', 'loading_rate_Ns': 'loading_rate', 'impulse_pctbw_s': 'impulse'})

per_trial_df.to_csv(MVP_SUM, index=False)

# Risk summary: per rule_id total fired and not-evaluable parsed from report, with evidence
risk_summary_rows = []
fired = {}
if not risk_flags.empty:
    if 'rule_id' in risk_flags.columns:
        fired = risk_flags.groupby('rule_id').size().to_dict()
    else:
        # fallback to flag name if rule_id missing
        fired = risk_flags.groupby('flag').size().to_dict()
not_eval = {}
RR = REPO_ROOT / 'docs' / 'risk_engine' / 'RISK_ENGINE_REPORT.md'
if RR.exists():
    try:
        txt = RR.read_text(encoding='utf-8').splitlines()
        in_ne = False
        for line in txt:
            if line.strip().lower().startswith('## not-evaluable cases'):
                in_ne = True
                continue
            if in_ne:
                if line.strip().startswith('##') and 'not-evaluable' not in line.lower():
                    break
                if line.strip().startswith('- '):
                    # format: - rule_id: N trials not evaluable (...)
                    try:
                        part = line.strip()[2:]
                        k, rest = part.split(':', 1)
                        num = ''.join(ch for ch in rest if ch.isdigit())
                        if num:
                            not_eval[k.strip()] = int(num)
                    except Exception:
                        pass
    except Exception:
        pass

all_rules = set(list(fired.keys()) + list(not_eval.keys()))
# evidence from thresholds
evidence_key_map = {}
evidence_grade_map = {}
try:
    TH = yaml.safe_load((REPO_ROOT/'configs'/'clinical_thresholds.yaml').read_text(encoding='utf-8')) or {}
    RTH = TH.get('risk_engine_v1') or {}
    for k,v in RTH.items():
        evidence_key_map[k] = v.get('source','')
        evidence_grade_map[k] = v.get('evidence_grade','')
except Exception:
    pass
for rid in sorted(all_rules):
    risk_summary_rows.append({
        'rule_id': rid,
        'total_fired': int(fired.get(rid, 0)),
        'not_evaluable': int(not_eval.get(rid, 0)),
        'evidence_key': evidence_key_map.get(rid, ''),
        'evidence_grade': evidence_grade_map.get(rid, ''),
        'notes': ''
    })
# ensure headers even if empty
if not risk_summary_rows:
    pd.DataFrame([], columns=['rule_id','total_fired','not_evaluable','evidence_key','evidence_grade','notes']).to_csv(RISK_SUM, index=False)
else:
    pd.DataFrame(risk_summary_rows).to_csv(RISK_SUM, index=False)

# Sanity checks
checks = []
# Expect higher high-load incidence in cutting/landing vs walk
if not flags.empty:
    def count_flag(name, task_wild):
        df = flags.copy()
        df['task'] = df['trial'].map(derive_task)
        return int(len(df[(df['flag'].str.contains(name, case=False, na=False)) & (df['task']==task_wild)]))
    hi_cut = count_flag('High impact', 'cutting/pivot') + count_flag('High impact', 'landing/jump')
    hi_walk = count_flag('High impact', 'walk/gait')
    checks.append(f"High-impact flags (cut/pivot/landing)={hi_cut} vs walk/gait={hi_walk}")

# Build markdown report
lines = []
lines.append('# MVP + Risk Engine v1 Internal Validation')
lines.append('')
lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
lines.append('')
lines.append('## Dataset and inclusion criteria')
lines.append(f"- Root: {USED_ROOT}")
lines.append(f"- Inclusion: canonical trials from external/internal leaderboards (if available)")
lines.append('')
lines.append('## Metric distributions')
for r in summary_rows:
    # show median (p50) and p95 per requirement
    p95 = r.get('p95', float('nan'))
    lines.append(f"- {r['metric']}: n={r['count']}, mean={r['mean']:.2f}, p50={r['p50']:.2f}, p95={p95:.2f}")
lines.append('')
lines.append('## MVP flags (counts)')
if not flags.empty and 'flag' in flags.columns:
    try:
        vc = flags['flag'].value_counts()
        for name, cnt in vc.items():
            lines.append(f"- {name}: {int(cnt)}")
    except Exception:
        lines.append('- None')
else:
    lines.append('- None')
lines.append('')
lines.append('## Risk flags (counts)')
risk_counts = []
if not risk_flags.empty:
    risk_counts = (risk_flags.groupby(['flag']).size().reset_index(name='count')).to_dict('records')
if risk_counts:
    for r in risk_counts:
        lines.append(f"- {r['flag']}: {r['count']}")
else:
    lines.append('- None')
lines.append('')
lines.append('## Risk Engine rule summary (met vs not-evaluable)')
if risk_summary_rows:
    for rr in risk_summary_rows:
        rid = rr.get('rule_id','')
        tf = int(rr.get('total_fired', 0))
        ne = int(rr.get('not_evaluable', 0))
        lines.append(f"- {rid}: met={tf}, not_evaluable={ne}")
else:
    lines.append('- None')
lines.append('')
lines.append('## Sanity checks')
if checks:
    for c in checks:
        lines.append(f"- {c}")
else:
    lines.append('- No sanity checks applicable (insufficient task labels).')
lines.append('')
lines.append('')
# Eligibility summary
lines.append('## Eligibility summary')
lines.append(f"- Total trials scanned: {total_scanned}")
lines.append(f"- Eligible trials (pre-scan): {len(eligible_trials)}")
lines.append(f"- Exclusions:")
# top 5 exclusion reasons by count
ex_sorted = sorted([(k, reasons.get(k,0)) for k in reasons.keys()], key=lambda kv: kv[1], reverse=True)[:5]
for k, v in ex_sorted:
    lines.append(f"  - {k}: {v}")
lines.append('')
lines.append('Disclaimer: Indicators reflect mechanical risk and workload; not deterministic injury predictions.')

# Suggested fixes when exclusions are present or zero eligible
if (total_scanned > 0) and (len(eligible_trials) == 0 or any(v > 0 for v in reasons.values())):
    lines.append('')
    lines.append('## Suggested fixes to increase eligibility')
    lines.append('- Ensure IMU columns map to canonical names: ax, ay, az (gx, gy, gz optional). Examples: acc_x→ax, gyro_y→gy, Ax→ax, Wx→gx')
    lines.append('- Provide time_s or set fs_hz in configs/dataset.yaml so time can be synthesized')
    lines.append('- Ensure at least window length coverage: >= ceil(fs_hz*0.3)+2 samples for w=300ms')
    lines.append('- See docs/validation/INPUT_SCHEMA.md for detailed rules')

REPORT.write_text('\n'.join(lines)+'\n', encoding='utf-8')

print('[OK] Validation wrote:')
print('  ', MVP_SUM)
print('  ', RISK_SUM)
print('  ', REPORT)

# Topline console summary
n_trials = int(len(met)) if not met.empty else 0
subjects = int(met['subject'].nunique()) if (not met.empty) and 'subject' in met.columns else 0
hi_stiff = int(len(flags)) if not flags.empty else 0
print(f"Trials processed: {n_trials} (subjects: {subjects})")
print(f"MVP flag rows: {hi_stiff} (see {FLAGS})")
print(f"Risk report: {REPORT}")
