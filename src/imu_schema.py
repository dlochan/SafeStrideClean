import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS = REPO_ROOT / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)
LOG = LOGS / 'validation_mvp_risk.log'

def _log(msg: str):
    try:
        with open(LOG, 'a', encoding='utf-8') as f:
            f.write('[IMU_SCHEMA] ' + msg + '\n')
    except Exception:
        pass


def normalize_imu_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize common IMU column variants to canonical names:
    - Ax,Ay,Az,Wx,Wy,Wz -> ax,ay,az,gx,gy,gz
    - acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z -> ax,ay,az,gx,gy,gz
    - Case-insensitive; strip spaces; convert '-' to '_'
    - Preserve sensor tags when present, e.g., ax_shank stays ax_shank

    Requires at least accelerometer axes (ax,ay,az) present in either single-sensor
    form or any single tag group (ax_<tag>, ay_<tag>, az_<tag>). Gyros are optional.
    Raises ValueError if accelerometer axes are still missing after mapping.
    """
    # Normalize raw column names
    df = df.copy()
    df.columns = [str(c).strip().replace(' ', '_').replace('-', '_') for c in df.columns]

    # Build rename map
    rename = {}
    for c in list(df.columns):
        lc = c.lower()
        # simple exacts
        if lc in ('ax','ay','az','gx','gy','gz','time_s'):
            continue
        # Camel-case Ax, Wx, etc.
        if lc in ('ax','ay','az','wx','wy','wz'):
            # handled above
            pass
        # Patterns: acc_x, accel_x, acceleration_x -> ax
        if lc.startswith('acc_') or lc.startswith('accel_') or lc.startswith('acceleration_') or lc.startswith('linearacc_'):
            axis = lc.split('_', 1)[1]
            if axis in ('x','y','z'):
                rename[c] = f"a{axis}"
                continue
        # Patterns: gyro_x, gyr_x -> gx
        if lc.startswith('gyro_') or lc.startswith('gyr_'):
            axis = lc.split('_', 1)[1]
            if axis in ('x','y','z'):
                rename[c] = f"g{axis}"
                continue
        # Compact ending: AccX, GyroY, etc.
        if lc.endswith('x') or lc.endswith('y') or lc.endswith('z'):
            head = lc[:-1]
            ax = lc[-1]
            if head in ('acc','accel','acceleration','linearacc'):
                rename[c] = f"a{ax}"
                continue
            if head in ('gyro','gyr'):
                rename[c] = f"g{ax}"
                continue
        # Uppercase Ax/Ay/Az/Wx/Wy/Wz
        if c in ('Ax','Ay','Az'):
            rename[c] = c.lower()
            continue
        if c in ('Wx','Wy','Wz'):
            # Wx.. -> gx..
            rename[c] = 'g' + c[1].lower()
            continue

    if rename:
        try:
            df = df.rename(columns=rename)
            _log(f"Column normalization: {rename}")
        except Exception:
            pass

    # Validate minimum accelerometer set exists
    base_acc = ['ax','ay','az']
    single_ok = all(b in df.columns for b in base_acc)
    multi_ok = False
    if not single_ok:
        # detect any complete 3-axis accelerometer set with suffix
        tags = {}
        for col in df.columns:
            for k in base_acc:
                pref = f"{k}_"
                if col.startswith(pref):
                    tag = col[len(pref):]
                    tags.setdefault(tag, set()).add(k)
        multi_ok = any(len(s) == 3 for s in tags.values())
    if not (single_ok or multi_ok):
        raise ValueError('bad_columns_missing_acc')

    return df


def ensure_time_s(df: pd.DataFrame, fs_hz: float | None) -> pd.DataFrame:
    """
    Ensure a monotonic 'time_s' column exists.
    - If present and monotonic non-decreasing: keep (sorted if necessary)
    - Else if fs_hz provided and >0: synthesize time_s = arange(n)/fs_hz
    - Else: raise ValueError('no_time_s_and_no_fs')
    """
    df = df.copy()
    if 'time_s' in df.columns:
        t = pd.to_numeric(df['time_s'], errors='coerce')
        if t.notna().sum() >= 2:
            try:
                if not t.is_monotonic_increasing:
                    df = df.assign(time_s=t).sort_values('time_s').reset_index(drop=True)
                return df
            except Exception:
                pass
    if fs_hz and float(fs_hz) > 0:
        n = len(df)
        df.insert(0, 'time_s', np.arange(n, dtype=float)/float(fs_hz))
        _log(f"synth_time_from_fs fs_hz={fs_hz}")
        return df
    raise ValueError('no_time_s_and_no_fs')
