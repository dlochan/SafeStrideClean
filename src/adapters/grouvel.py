import os, csv, json, re
from pathlib import Path
import pandas as pd
import numpy as np

def _read_truth_mapping(cfg_path: Path) -> dict:
    m = {
        'ap_axis': 'Fx', 'ml_axis': 'Fy', 'vz_axis': 'Fz',
        'ap_sign': 1, 'ml_sign': 1, 'vz_sign': 1,
    }
    try:
        if not cfg_path.exists():
            return m
        for line in cfg_path.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line.startswith('ap_axis:'):
                m['ap_axis'] = line.split(':',1)[1].strip()
            elif line.startswith('ml_axis:'):
                m['ml_axis'] = line.split(':',1)[1].strip()
            elif line.startswith('vz_axis:'):
                m['vz_axis'] = line.split(':',1)[1].strip()
            elif line.startswith('ap_sign:'):
                m['ap_sign'] = int(line.split(':',1)[1].strip())
            elif line.startswith('ml_sign:'):
                m['ml_sign'] = int(line.split(':',1)[1].strip())
            elif line.startswith('vz_sign:'):
                m['vz_sign'] = int(line.split(':',1)[1].strip())
    except Exception:
        pass
    return m

def _apply_truth_mapping(df: pd.DataFrame) -> dict:
    """Return dict of arrays for Fx_N, Fy_N, Fz_N using mapping and signs if present."""
    cfg = Path(r"C:\Users\locha\Documents\safestride\configs\dataset.yaml")
    mp = _read_truth_mapping(cfg)
    def pick(colname: str) -> pd.Series | None:
        # prefer exact colname match (case-insensitive)
        for c in df.columns:
            if str(c).lower() == colname.lower() or re.search(fr'(^|_){colname.lower()}(_|$)', str(c).lower()):
                return pd.to_numeric(df[c], errors='coerce')
        return None
    comp = {
        'Fx_N': pick(mp['ap_axis']) if mp.get('ap_axis') else None,
        'Fy_N': pick(mp['ml_axis']) if mp.get('ml_axis') else None,
        'Fz_N': pick(mp['vz_axis']) if mp.get('vz_axis') else None,
    }
    # fallback heuristics if any missing
    if comp['Fz_N'] is None:
        comp['Fz_N'] = _extract_force_component(df, 'Fz')
    if comp['Fx_N'] is None:
        comp['Fx_N'] = _extract_force_component(df, 'Fx')
    if comp['Fy_N'] is None:
        comp['Fy_N'] = _extract_force_component(df, 'Fy')
    # apply signs
    if comp['Fx_N'] is not None:
        comp['Fx_N'] = comp['Fx_N'] * float(mp.get('ap_sign', 1))
    if comp['Fy_N'] is not None:
        comp['Fy_N'] = comp['Fy_N'] * float(mp.get('ml_sign', 1))
    if comp['Fz_N'] is not None:
        comp['Fz_N'] = comp['Fz_N'] * float(mp.get('vz_sign', 1))
    return comp

"""
Lightweight helpers to probe and (where already canonical) standardize Grouvel dataset files.
If non-canonical Excel formats are encountered, this module records them in the mapping report
for later parser buildout. No schema-heavy parsing is attempted without samples.
"""

def find_candidate_files(data_root: Path) -> dict:
    out = {
        'inertial_xlsx': [],
        'insoles_xlsx': [],
        'opto_xlsx': [],
        'imu_csv': [],
        'grf_csv': [],
        'insoles_csv': [],
    }
    for p in data_root.rglob('*'):
        if not p.is_file():
            continue
        n = p.name.lower()
        if n.endswith('.xlsx') and 'inertial' in n:
            out['inertial_xlsx'].append(p)
        elif n.endswith('.xlsx') and ('insoles' in n or 'insole' in n):
            out['insoles_xlsx'].append(p)
        elif n.endswith('.xlsx') and ('opto' in n or 'optoelectronic' in n or 'marker' in n):
            out['opto_xlsx'].append(p)
        elif n.endswith('.csv') and n.endswith('_imu_real.csv'):
            out['imu_csv'].append(p)
        elif n.endswith('.csv') and n.endswith('_grf.csv'):
            out['grf_csv'].append(p)
        elif n.endswith('.csv') and 'insole' in n:
            out['insoles_csv'].append(p)
    return out


# --- New: Parse SYNC_DATA CSVs into canonical trio ---

TIME_CANDIDATES = [
    re.compile(r'^(time(_s)?|timestamp|time_ms|t)$', re.I)
]

def _find_time_col(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        for rx in TIME_CANDIDATES:
            if rx.match(str(c)):
                return c
    # fallback: first numeric monotonic
    for c in df.columns:
        s = pd.to_numeric(df[c], errors='coerce')
        if s.notna().sum() > 50:
            vals = s.dropna().values[:300]
            if vals.size > 3 and np.all(np.diff(vals) > 0):
                return c
    return None

def _extract_force_component(df: pd.DataFrame, name: str) -> pd.Series | None:
    # name like 'Fx','Fy','Fz'; accept variations and units in N
    for c in df.columns:
        lc = str(c).lower()
        if re.search(fr'(^|_){name.lower()}(_|$)', lc):
            return pd.to_numeric(df[c], errors='coerce')
    # attempt generic labels
    if name.lower()=='fz':
        return _extract_fz(df)
    return None

def _find_imu_axes(df: pd.DataFrame) -> dict:
    """
    Heuristics: group columns into sensors by detecting patterns like
    (acc|imu).*(_?x|_?y|_?z) and (gyro|gyr).*(_?x|_?y|_?z), with sensor tag
    tokens containing thigh/shank/tibia/femur or r/l prefixes.
    Return mapping tag -> dict(ax,ay,az,gx,gy,gz) as Series.
    """
    cols = list(map(str, df.columns))
    candidates = {}
    for c in cols:
        lc = c.lower()
        # extract tag
        # pattern 1: verbose tag names (thigh, shank, tibia, femur, pelvis, knee)
        m_acc = re.search(r'(acc|imu).*?([rl]?(?:thigh|shank|tibia|femur|pelvis|knee)[^_\-]*)', lc)
        m_gyr = re.search(r'(gyr|gyro).*?([rl]?(?:thigh|shank|tibia|femur|pelvis|knee)[^_\-]*)', lc)
        # pattern 2: P6_LT_acc_x / P6_LS_acc_x / P6_RT_* / P6_RS_*
        if not (m_acc or m_gyr):
            m_acc = re.search(r'p\d+_([lr](?:t|s))_acc_([xyz])', lc)
            m_gyr = re.search(r'p\d+_([lr](?:t|s))_gyro_([xyz])', lc)
        def ax_letter(name: str):
            m = re.search(r'([^a-z]|_)([xyz])([^a-z]|$)', name)
            return m.group(2) if m else None
        if m_acc:
            tag = m_acc.group(1) if m_acc.lastindex and m_acc.lastindex>=1 and len(m_acc.groups())==2 else m_acc.group(2)
            a = m_acc.group(2) if m_acc.lastindex and m_acc.lastindex>=2 and len(m_acc.groups())==2 else ax_letter(lc)
            if a:
                candidates.setdefault(tag, {}).setdefault('acc', {})[a] = c
        if m_gyr:
            tag = m_gyr.group(1) if m_gyr.lastindex and m_gyr.lastindex>=1 and len(m_gyr.groups())==2 else m_gyr.group(2)
            a = m_gyr.group(2) if m_gyr.lastindex and m_gyr.lastindex>=2 and len(m_gyr.groups())==2 else ax_letter(lc)
            if a:
                candidates.setdefault(tag, {}).setdefault('gyr', {})[a] = c
    out = {}
    for tag, parts in candidates.items():
        acc = parts.get('acc', {})
        gyr = parts.get('gyr', {})
        if set(acc.keys()) >= {'x','y','z'} and set(gyr.keys()) >= {'x','y','z'}:
            # normalize tag
            norm = tag.replace('right','r').replace('left','l')
            norm = norm.replace('tibia','shank').replace('femur','thigh')
            norm = norm.replace('knee','thigh')
            # map compact tags: lt->lpthigh, ls->lshank, rt->rpthigh, rs->rshank
            if norm in ('lt','l t'):
                norm = 'lpthigh'
            elif norm in ('ls','l s'):
                norm = 'lshank'
            elif norm in ('rt','r t'):
                norm = 'rpthigh'
            elif norm in ('rs','r s'):
                norm = 'rshank'
            out[norm] = {
                'ax': df[acc['x']], 'ay': df[acc['y']], 'az': df[acc['z']],
                'gx': df[gyr['x']], 'gy': df[gyr['y']], 'gz': df[gyr['z']],
            }
    return out

def _choose_knee_pair(tags: list[str]) -> tuple[str,str] | None:
    # prefer rthigh/rshank else lthigh/lshank else any thigh/shank pair
    tset = set(tags)
    for side in ['l','r']:
        thigh = f"{side}thigh" if f"{side}thigh" in tset else f"{side}pthigh" if f"{side}pthigh" in tset else None
        shank = f"{side}shank" if f"{side}shank" in tset else None
        if thigh and shank:
            return thigh, shank
    # generic
    thigh = next((t for t in tags if 'thigh' in t), None)
    shank = next((t for t in tags if 'shank' in t), None)
    if thigh and shank:
        return thigh, shank
    return None

def _extract_fz(df: pd.DataFrame) -> pd.Series | None:
    # try direct vertical force
    for c in df.columns:
        lc = str(c).lower()
        if re.search(r'(^|_)fz(_|$)', lc) or ('vertical' in lc and 'force' in lc):
            s = pd.to_numeric(df[c], errors='coerce')
            return s
    # insole derived: sum columns matching (insole.*force|pressure) and assume unit N if already force
    cand = [c for c in df.columns if re.search(r'(insole|pressure).*force', str(c).lower())]
    if cand:
        S = sum(pd.to_numeric(df[c], errors='coerce').fillna(0.0) for c in cand)
        return S
    return None

def standardize_sync_trial(sync_csv: Path, work_root: Path) -> dict | None:
    # Robust delimiter/decimal sniff
    try:
        try:
            df = pd.read_csv(sync_csv, engine='python', sep=None)
        except Exception:
            df = pd.read_csv(sync_csv)
    except Exception:
        return None
    tcol = _find_time_col(df)
    imu = _find_imu_axes(df)
    if not imu:
        return None
    pair = _choose_knee_pair(list(imu.keys()))
    if not pair:
        return None
    thigh, shank = pair
    # compose canonical IMU
    out_rows = []
    n = len(df)
    if tcol is None:
        t = pd.Series(np.arange(len(df))/100.0)
    else:
        t = pd.to_numeric(df[tcol], errors='coerce')
        if t.isna().all():
            # fallback to index-based time at 100 Hz
            n = len(df)
            t = pd.Series(np.arange(n)/100.0)
        else:
            if t.dropna().median() > 10:  # ms
                t = t/1000.0
    d_can = pd.DataFrame({
        'time_s': t,
        f'ax_{thigh}': pd.to_numeric(imu[thigh]['ax'], errors='coerce'),
        f'ay_{thigh}': pd.to_numeric(imu[thigh]['ay'], errors='coerce'),
        f'az_{thigh}': pd.to_numeric(imu[thigh]['az'], errors='coerce'),
        f'gx_{thigh}': pd.to_numeric(imu[thigh]['gx'], errors='coerce'),
        f'gy_{thigh}': pd.to_numeric(imu[thigh]['gy'], errors='coerce'),
        f'gz_{thigh}': pd.to_numeric(imu[thigh]['gz'], errors='coerce'),
        f'ax_{shank}': pd.to_numeric(imu[shank]['ax'], errors='coerce'),
        f'ay_{shank}': pd.to_numeric(imu[shank]['ay'], errors='coerce'),
        f'az_{shank}': pd.to_numeric(imu[shank]['az'], errors='coerce'),
        f'gx_{shank}': pd.to_numeric(imu[shank]['gx'], errors='coerce'),
        f'gy_{shank}': pd.to_numeric(imu[shank]['gy'], errors='coerce'),
        f'gz_{shank}': pd.to_numeric(imu[shank]['gz'], errors='coerce'),
    })
    trial = sync_csv.stem
    work_root.mkdir(parents=True, exist_ok=True)
    imu_out = work_root / f"{trial}_imu_real.csv"
    if not imu_out.exists():
        d_can.to_csv(imu_out, index=False)
    # Forces
    comp = _apply_truth_mapping(df)
    fz = comp.get('Fz_N'); fx = comp.get('Fx_N'); fy = comp.get('Fy_N')
    grf_out = None
    if fz is not None:
        grf_out = work_root / f"{trial}_grf.csv"
        if not grf_out.exists():
            pd.DataFrame({'time_s': d_can['time_s'], 'Fz_N': pd.to_numeric(fz, errors='coerce')}).to_csv(grf_out, index=False)
    # 3D truth if available
    if any(s is not None for s in [fx, fy, fz]):
        truth3d = work_root / f"{trial}_grf3d_truth.csv"
        if not truth3d.exists():
            dd = {'time_s': d_can['time_s']}
            if fx is not None: dd['Fx_N'] = pd.to_numeric(fx, errors='coerce')
            if fy is not None: dd['Fy_N'] = pd.to_numeric(fy, errors='coerce')
            if fz is not None: dd['Fz_N'] = pd.to_numeric(fz, errors='coerce')
            pd.DataFrame(dd).to_csv(truth3d, index=False)
    # Activity flag (simple placeholder; downstream tools may refine)
    act_out = work_root / f"{trial}_activity_flag.csv"
    if not act_out.exists():
        pd.DataFrame({'active': [1]}).to_csv(act_out, index=False)
    return {
        'trial': trial,
        'imu_out': str(imu_out),
        'grf_out': str(grf_out) if grf_out else '',
        'activity_out': str(act_out),
        'thigh': thigh,
        'shank': shank,
    }


def infer_trial_id(path: Path) -> str:
    s = path.stem
    for suf in ['_imu_real','_grf','_insoles','_insoles_force','_insoles_pressure']:
        if s.endswith(suf):
            return s[:-len(suf)]
    return s


def summarize_streams(files: dict) -> pd.DataFrame:
    rows = []
    for k, lst in files.items():
        for p in lst:
            rows.append({'stream': k, 'path': str(p)})
    return pd.DataFrame(rows)


def standardize_canonical_trial(imu_csv: Path, grf_src: Path | None, work_root: Path) -> dict:
    trial = infer_trial_id(imu_csv)
    out = {
        'trial': trial,
        'imu_out': str(work_root / f"{trial}_imu_real.csv"),
        'grf_out': None,
        'activity_out': str(work_root / f"{trial}_activity_flag.csv"),
        'notes': '',
    }
    # Copy IMU canonical file
    work_root.mkdir(parents=True, exist_ok=True)
    if not Path(out['imu_out']).exists():
        pd.read_csv(imu_csv).to_csv(out['imu_out'], index=False)
    # GRF if provided as canonical
    if grf_src and grf_src.exists():
        out['grf_out'] = str(work_root / f"{trial}_grf.csv")
        if not Path(out['grf_out']).exists():
            g = pd.read_csv(grf_src)
            # ensure Fz_N column present or convertible
            if 'Fz_N' in g.columns:
                g[['time_s','Fz_N']].to_csv(out['grf_out'], index=False)
            else:
                g.to_csv(out['grf_out'], index=False)
            # try to emit 3D truth as well if Fx/Fy present
            comp = _apply_truth_mapping(g)
            fx = comp.get('Fx_N'); fy = comp.get('Fy_N'); fz = comp.get('Fz_N')
            if any(s is not None for s in [fx, fy, fz]):
                truth3d = work_root / f"{trial}_grf3d_truth.csv"
                if not truth3d.exists():
                    dd = {'time_s': pd.to_numeric(g.get('time_s', pd.Series(range(len(g)))/200.0), errors='coerce')}
                    if fx is not None: dd['Fx_N'] = pd.to_numeric(fx, errors='coerce')
                    if fy is not None: dd['Fy_N'] = pd.to_numeric(fy, errors='coerce')
                    if fz is not None: dd['Fz_N'] = pd.to_numeric(fz, errors='coerce')
                    pd.DataFrame(dd).to_csv(truth3d, index=False)
    # simple activity flag if missing later
    if not Path(out['activity_out']).exists():
        pd.DataFrame({'active':[1]}).to_csv(out['activity_out'], index=False)
    return out
