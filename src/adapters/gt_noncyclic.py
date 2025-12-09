# src/adapters/gt_noncyclic.py
import pandas as pd
import numpy as np
from typing import List, Optional, Dict

def _norm(cols):
    out = []
    for c in cols:
        c2 = c.strip().lower()
        if "(" in c2 and ")" in c2:
            c2 = c2[:c2.index("(")].strip()
        c2 = c2.replace(" ", "_")
        out.append(c2)
    return out

def _maybe_gravity_remove(df: pd.DataFrame, cols: List[str], mode: str = "none") -> pd.DataFrame:
    """Remove gravity from accelerometer channels.
    mode: 'none' | 'mean'. If 'mean', subtract column mean.
    """
    if mode is None or str(mode).lower() == "none":
        return df
    out = df.copy()
    if str(mode).lower() == "mean":
        for c in cols:
            if c in out.columns:
                mu = np.nanmean(out[c].to_numpy(dtype=float))
                out[c] = out[c].astype(float) - (0.0 if np.isnan(mu) else mu)
        return out
    # Fallback: no-op if unknown
    return out

def _resolve_col(df_cols: List[str], base: str, alts: List[str]) -> Optional[str]:
    """Return the first matching column name among alternatives present in df_cols.
    Example: base='lshank', alts=['accx','ax','acc_x'] -> tries 'lshank_accx', ...
    """
    for a in alts:
        cand = f"{base}_{a}"
        if cand in df_cols:
            return cand
    return None

# ------------------ IMU (single sensor) ------------------
def load_gt_imu_real(path_csv: str, sensor: str = "rshank", gravity_remove: str = "none") -> pd.DataFrame:
    """
    Returns columns: ['time_s','ax','ay','az','gx','gy','gz'] for ONE sensor.
    """
    df = pd.read_csv(path_csv)
    df.columns = _norm(df.columns)
    if "time" not in df.columns and "time_s" not in df.columns:
        raise ValueError(f"[GT-IMU] No time column in {path_csv}. Got: {df.columns.tolist()}")
    time_col = "time_s" if "time_s" in df.columns else "time"

    # Try robust resolution of column names
    acc_alts = ["accx", "ax", "acc_x"]
    acy_alts = ["accy", "ay", "acc_y"]
    acz_alts = ["accz", "az", "acc_z"]
    grx_alts = ["gyrox", "gx", "gyro_x"]
    gry_alts = ["gyroy", "gy", "gyro_y"]
    grz_alts = ["gyroz", "gz", "gyro_z"]

    col_ax = _resolve_col(df.columns, sensor, acc_alts)
    col_ay = _resolve_col(df.columns, sensor, acy_alts)
    col_az = _resolve_col(df.columns, sensor, acz_alts)
    col_gx = _resolve_col(df.columns, sensor, grx_alts)
    col_gy = _resolve_col(df.columns, sensor, gry_alts)
    col_gz = _resolve_col(df.columns, sensor, grz_alts)
    cols_map: Dict[str, Optional[str]] = {"ax": col_ax, "ay": col_ay, "az": col_az, "gx": col_gx, "gy": col_gy, "gz": col_gz}
    missing = [k for k, v in cols_map.items() if v is None]
    if missing:
        raise ValueError(
            f"[GT-IMU] Missing columns for sensor '{sensor}' ({missing}). Available: {df.columns.tolist()}"
        )

    out = pd.DataFrame({
        "time_s": df[time_col].astype(float),
        "ax": df[col_ax].astype(float),
        "ay": df[col_ay].astype(float),
        "az": df[col_az].astype(float),
        "gx": df[col_gx].astype(float),
        "gy": df[col_gy].astype(float),
        "gz": df[col_gz].astype(float),
    })
    out = _maybe_gravity_remove(out, ["ax","ay","az"], gravity_remove)
    return out.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)

# ------------------ IMU (multi sensor) ------------------
def load_gt_imu_multi(path_csv: str, sensors: List[str], gravity_remove: str = "none") -> pd.DataFrame:
    """
    Returns columns:
      ['time_s',
       ax_<s1>, ay_<s1>, az_<s1>, gx_<s1>, gy_<s1>, gz_<s1>,
       ax_<s2>, ay_<s2>, ...]  (for all sensors in order)
    """
    df = pd.read_csv(path_csv)
    df.columns = _norm(df.columns)
    if "time" not in df.columns and "time_s" not in df.columns:
        raise ValueError(f"[GT-IMU] No time column in {path_csv}. Got: {df.columns.tolist()}")
    time_col = "time_s" if "time_s" in df.columns else "time"

    out = pd.DataFrame({"time_s": df[time_col].astype(float)})
    for s in sensors:
        # resolve robustly
        ax = _resolve_col(df.columns, s, ["accx", "ax", "acc_x"]) 
        ay = _resolve_col(df.columns, s, ["accy", "ay", "acc_y"]) 
        az = _resolve_col(df.columns, s, ["accz", "az", "acc_z"]) 
        gx = _resolve_col(df.columns, s, ["gyrox", "gx", "gyro_x"]) 
        gy = _resolve_col(df.columns, s, ["gyroy", "gy", "gyro_y"]) 
        gz = _resolve_col(df.columns, s, ["gyroz", "gz", "gyro_z"]) 
        need_map = {"ax": ax, "ay": ay, "az": az, "gx": gx, "gy": gy, "gz": gz}
        miss = [k for k, v in need_map.items() if v is None]
        if miss:
            raise ValueError(f"[GT-IMU] Missing columns for sensor '{s}' ({miss}). Available: {df.columns.tolist()}")
        out[f"ax_{s}"] = df[ax].astype(float)
        out[f"ay_{s}"] = df[ay].astype(float)
        out[f"az_{s}"] = df[az].astype(float)
        out[f"gx_{s}"] = df[gx].astype(float)
        out[f"gy_{s}"] = df[gy].astype(float)
        out[f"gz_{s}"] = df[gz].astype(float)

    # optional gravity removal per sensor's acc channels
    if gravity_remove and gravity_remove.lower() != "none":
        cols = [c for c in out.columns if c.startswith("a") and c != "time_s"]
        out = _maybe_gravity_remove(out, cols, gravity_remove)

    return out.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)

# ------------------ GRF total ------------------
def load_gt_grf_total(path_csv: str) -> pd.DataFrame:
    """
    Total GRF across both feet -> ['time_s','Fx_N','Fy_N','Fz_N'].
    Y is vertical in this dataset: Fz_N = rforcey_vertical + lforcey_vertical
    """
    df = pd.read_csv(path_csv)
    df.columns = _norm(df.columns)
    if "time" not in df.columns and "time_s" not in df.columns:
        raise ValueError(f"[GT-GRF] No time column in {path_csv}. Got: {df.columns.tolist()}")
    time_col = "time_s" if "time_s" in df.columns else "time"

    def get(c): return df[c].astype(float) if c in df.columns else 0.0

    rfx = np.nan_to_num(get("rforcex"), nan=0.0)
    rfy = np.nan_to_num(get("rforcey_vertical"), nan=0.0)  # vertical
    rfz = np.nan_to_num(get("rforcez"), nan=0.0)
    lfx = np.nan_to_num(get("lforcex"), nan=0.0)
    lfy = np.nan_to_num(get("lforcey_vertical"), nan=0.0)  # vertical
    lfz = np.nan_to_num(get("lforcez"), nan=0.0)

    out = pd.DataFrame({
        "time_s": df[time_col].astype(float),
        "Fx_N": rfx + lfx,   # AP
        "Fy_N": rfz + lfz,   # ML
        "Fz_N": rfy + lfy,   # Vertical
    })
    return out.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)

# ------------------ Activity flag & filtering ------------------
def load_gt_activity_flag(path_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_csv)
    df.columns = _norm(df.columns)
    time_col = "time_s" if "time_s" in df.columns else ("time" if "time" in df.columns else None)
    if time_col is None:
        raise ValueError(f"[GT-FLAG] No time column in {path_csv}")
    flag_col = None
    for c in df.columns:
        if c == time_col:
            continue
        series = df[c].dropna()
        if len(series) > 0 and series.isin([0,1,True,False]).all():
            flag_col = c
            break
    if flag_col is None:
        raise ValueError(f"[GT-FLAG] Could not find a boolean flag column in {path_csv}")
    return pd.DataFrame({"time_s": df[time_col].astype(float), "active": df[flag_col].astype(int)})

def filter_active(imu_df: pd.DataFrame, grf_df: pd.DataFrame, flag_df: pd.DataFrame):
    m_imu = imu_df.merge(flag_df, on="time_s", how="inner")
    m_grf = grf_df.merge(flag_df, on="time_s", how="inner")
    m_imu = m_imu[m_imu["active"] == 1].drop(columns=["active"]).reset_index(drop=True)
    m_grf = m_grf[m_grf["active"] == 1].drop(columns=["active"]).reset_index(drop=True)
    return m_imu, m_grf
