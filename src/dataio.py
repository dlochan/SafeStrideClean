import pandas as pd
import numpy as np
import ezc3d
import os
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGS = REPO_ROOT / 'logs'
LOGS.mkdir(parents=True, exist_ok=True)
_LOG_PATH = LOGS / 'timeseries_loader.log'

def _log_once_for(path_str: str, fs_hz: float):
    try:
        with open(_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(f"[DATAIO] synth_time_from_fs: path={path_str} fs_hz={fs_hz}\n")
    except Exception:
        pass

def _get_fs_hz_default() -> float:
    # read from configs/dataset.yaml if available
    try:
        cfg = REPO_ROOT / 'configs' / 'dataset.yaml'
        if cfg.exists():
            import yaml
            d = yaml.safe_load(cfg.read_text(encoding='utf-8')) or {}
            v = float(d.get('fs_hz', 200.0))
            if v and v > 0:
                return v
    except Exception:
        pass
    return 200.0


def load_imu_csv(path: str) -> pd.DataFrame:
    """
    Load IMU CSV file and validate its columns.

    Parameters:
    - path: str : Path to the IMU CSV file.

    Returns:
    - pd.DataFrame : DataFrame containing ['time_s', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']

    Raises:
    - ValueError: If columns are missing or time is not strictly increasing.
    """
    df = pd.read_csv(path)
    # normalize column names: case-insensitive, unify separators
    orig_cols = list(df.columns)
    cols_norm = [str(c).strip().lower().replace(' ', '_').replace('-', '_') for c in orig_cols]
    df.columns = cols_norm
    # time_s fallback if missing
    if 'time_s' not in df.columns:
        fs = _get_fs_hz_default()
        n = len(df)
        df.insert(0, 'time_s', np.arange(n, dtype=float) / fs)
        _log_once_for(str(path), fs)
    # Axis variant normalization to canonical names (preserve suffix tag)
    # Map variants: Acc/Accel/LinearAcc -> a{axis}; Gyro/Gyr -> g{axis}
    # Examples: acc_x -> ax, accel_y -> ay, linearacc_z -> az, gyro_x -> gx, gyr_y -> gy, gyrz -> gz
    rename_map = {}
    base_axes = {'x':'x','y':'y','z':'z'}
    for c in list(df.columns):
        if c == 'time_s':
            continue
        cl = c
        # detect axis letter and optional suffix tag
        axis = None
        tag = None
        # common patterns with underscores
        for ax in ('x','y','z'):
            if cl.endswith(f'_{ax}'):
                head = cl[:-(len(ax)+1)]
                axis = ax
                tag = None
                typ = head
                break
            if f'_{ax}_' in cl:
                # split into type + tag
                parts = cl.split(f'_{ax}_', 1)
                typ = parts[0]
                axis = ax
                tag = parts[1]
                break
        if axis is None:
            # no underscores; try compact forms
            for ax in ('x','y','z'):
                if cl.endswith(ax):
                    head = cl[:-1]
                    typ = head.rstrip('_')
                    axis = ax
                    tag = None
                    break
        if axis is None:
            continue
        typ_l = typ.replace('_', '')
        is_acc = any(k in typ_l for k in ['accel','linearacc','acceleration','acc']) and not any(k in typ_l for k in ['gyro','gyr'])
        is_gyr = any(k in typ_l for k in ['gyro','gyr'])
        if not (is_acc or is_gyr):
            continue
        new_base = ('a' if is_acc else 'g') + axis
        new_name = new_base if not tag else f"{new_base}_{tag}"
        if new_name != c:
            rename_map[c] = new_name
    if rename_map:
        try:
            df = df.rename(columns=rename_map)
            # log mapping once
            with open(_LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(f"[DATAIO] Column normalization {path}: {rename_map}\n")
        except Exception:
            pass
    # Accept either single-sensor (ax..gz) or multi-sensor ax_<tag>...gz_<tag>
    base = ['ax','ay','az','gx','gy','gz']
    single_ok = all(c in df.columns for c in base)
    multi_ok = False
    if not single_ok:
        # detect any complete 6-axis set with suffix
        tags = {}
        for c in df.columns:
            for k in base:
                pref = f"{k}_"
                if str(c).startswith(pref):
                    tag = str(c)[len(pref):]
                    tags.setdefault(tag, set()).add(k)
        multi_ok = any(len(s) == 6 for s in tags.values())
    if not (single_ok or multi_ok):
        raise ValueError("Missing IMU columns: require ax..gz or ax_<tag>..gz_<tag> sets")

    if not df['time_s'].is_monotonic_increasing:
        # if non-monotonic, attempt to sort by time; if still bad, raise
        try:
            df = df.sort_values('time_s').reset_index(drop=True)
        except Exception:
            raise ValueError("Time column must be strictly increasing")

    return df


def load_grf_csv(path: str) -> pd.DataFrame:
    """
    Load GRF CSV file and validate its columns.

    Parameters:
    - path: str : Path to the GRF CSV file.

    Returns:
    - pd.DataFrame : DataFrame containing ['time_s', 'Fx_N', 'Fy_N', 'Fz_N']

    Raises:
    - ValueError: If columns are missing.
    """
    df = pd.read_csv(path)
    required_columns = ['time_s', 'Fx_N', 'Fy_N', 'Fz_N']
    if not all(column in df.columns for column in required_columns):
        raise ValueError(f"Missing columns in GRF CSV, required: {required_columns}")

    return df


def load_c3d_grf(path: str) -> pd.DataFrame:
    """
    Load C3D file and extract first plate forces.

    Parameters:
    - path: str : Path to the C3D file.

    Returns:
    - pd.DataFrame : DataFrame containing extracted forces with time_s built from analog rate.
    """
    c3d = ezc3d.c3d(path)
    analog_data = c3d['data']['analogs'][0]
    analog_rate = c3d['parameters']['ANALOG']['RATE']['value'][0]
    time_s = np.arange(analog_data.shape[1]) / analog_rate
    df = pd.DataFrame(analog_data.T, columns=['Fx_N', 'Fy_N', 'Fz_N'])
    df.insert(0, 'time_s', time_s)
    return df


def resample_grf_to_imu_time(imu_df: pd.DataFrame, grf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample GRF data to match IMU time using forward-fill (ZOH).

    Parameters:
    - imu_df: pd.DataFrame : DataFrame containing IMU data.
    - grf_df: pd.DataFrame : DataFrame containing GRF data.

    Returns:
    - pd.DataFrame : Resampled GRF DataFrame.
    """
    grf_resampled = grf_df.set_index('time_s').reindex(imu_df['time_s'], method='ffill').reset_index()
    return grf_resampled


def save_predicted_grf(time_s: np.ndarray, Fz_BW: np.ndarray, bw_kg: float, out_csv: str):
    """
    Save predicted GRF to CSV file.

    Parameters:
    - time_s: np.ndarray : Array of time steps.
    - Fz_BW: np.ndarray : Normalized vertical GRF.
    - bw_kg: float : Body weight in kilograms.
    - out_csv: str : Output CSV file path.
    """
    Fz_N = Fz_BW * bw_kg * 9.81
    df = pd.DataFrame({'time_s': time_s, 'Fx_N': np.zeros_like(Fz_N), 'Fy_N': np.zeros_like(Fz_N), 'Fz_N': Fz_N})
    df.to_csv(out_csv, index=False)


# Optional vNext dataset aliases. These simply re-export the canonical vNext
# dataset classes when vNext is available, without introducing any new
# dependency from vNext back into dataio.
try:
    from vnext.data.datasets import DualIMUTrialDataset as _DualIMUTrialDataset
    from vnext.data.datasets import WindowedIMUDataset as _WindowedIMUDataset

    DualIMUTrialDataset = _DualIMUTrialDataset
    WindowedIMUDataset = _WindowedIMUDataset
except Exception:
    # vNext not available in this environment; core dataio functionality
    # remains usable without the vNext-specific dataset classes.
    pass


if __name__ == "__main__":
    sample_imu_path = "sample_imu.csv"
    sample_grf_path = "sample_grf.csv"
    sample_c3d_path = "sample.c3d"

    if os.path.exists(sample_imu_path):
        imu_df = load_imu_csv(sample_imu_path)
        print("IMU Data Head:")
        print(imu_df.head())

    if os.path.exists(sample_grf_path):
        grf_df = load_grf_csv(sample_grf_path)
        print("GRF Data Head:")
        print(grf_df.head())

    if os.path.exists(sample_c3d_path):
        c3d_df = load_c3d_grf(sample_c3d_path)
        print("C3D GRF Data Head:")
        print(c3d_df.head())
