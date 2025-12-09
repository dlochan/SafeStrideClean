import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_calib(s: str):
    # formats: 'standing=auto' or 'file=path.json'
    if not s:
        return {"mode": "auto"}
    parts = str(s)
    if parts.startswith("standing="):
        return {"mode": parts.split("=",1)[1]}
    if parts.startswith("file="):
        return {"mode": "file", "path": parts.split("=",1)[1]}
    return {"mode": "auto"}


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser(description="Estimate simple kinematics from dual IMU CSV")
    ap.add_argument("--imu_csv", required=True)
    ap.add_argument("--sensors", default="lpthigh,lshank")
    ap.add_argument("--fs", type=float, default=200.0)
    ap.add_argument("--calib", default="standing=auto")
    ap.add_argument("--out_prefix", required=True)
    ap.add_argument("--log_file", default=None)
    args = ap.parse_args()

    # Load IMU
    imu = pd.read_csv(args.imu_csv)
    if "time_s" not in imu.columns:
        raise SystemExit("imu_csv missing time_s column")
    t = imu["time_s"].to_numpy()
    dt = np.median(np.diff(t)) if len(t) > 1 else 1.0/args.fs

    # Identify sensor column suffixes
    # Expect columns like gx_thigh, gx_shank, gy_* gz_*
    def find_col(prefix: str, tag: str):
        for c in imu.columns:
            if c.startswith(prefix) and c.endswith(tag):
                return c
        # fallback common names
        name = f"{prefix}_{tag}"
        return name if name in imu.columns else None

    # Angular velocities (deg/s) proxies
    gy_thigh = imu.get("gy_thigh")
    gy_shank = imu.get("gy_shank")
    gx_thigh = imu.get("gx_thigh")
    gx_shank = imu.get("gx_shank")
    gz_thigh = imu.get("gz_thigh")
    gz_shank = imu.get("gz_shank")

    # If missing, try alternative prefixes
    if gy_thigh is None:
        gy_thigh = imu.filter(like="gy_").filter(like="thigh").iloc[:,0] if any(imu.columns.str.contains("gy_")) else pd.Series(np.zeros(len(t)))
    if gy_shank is None:
        gy_shank = imu.filter(like="gy_").filter(like="shank").iloc[:,0] if any(imu.columns.str.contains("gy_")) else pd.Series(np.zeros(len(t)))
    if gx_thigh is None:
        gx_thigh = imu.filter(like="gx_").filter(like="thigh").iloc[:,0] if any(imu.columns.str.contains("gx_")) else pd.Series(np.zeros(len(t)))
    if gx_shank is None:
        gx_shank = imu.filter(like="gx_").filter(like="shank").iloc[:,0] if any(imu.columns.str.contains("gx_")) else pd.Series(np.zeros(len(t)))

    omega_thigh = gy_thigh.to_numpy(dtype=float)
    omega_shank = gy_shank.to_numpy(dtype=float)
    omega_knee = omega_thigh - omega_shank

    # Integrate omega to angle (deg) with simple cumulative trapezoid
    ang = np.cumsum((omega_knee[:-1] + omega_knee[1:]) * 0.5 * dt)
    ang = np.concatenate([[0.0], ang])
    # detrend simple drift by subtracting linear trend
    if len(ang) > 1:
        slope = (ang[-1] - ang[0]) / max(len(ang)-1, 1)
        ang = ang - slope * np.arange(len(ang))

    # Frontal proxy from gx difference
    front = (np.asarray(gx_thigh, float) - np.asarray(gx_shank, float)).reshape(-1)
    # Events: simple threshold on shank omega magnitude
    wmag = np.abs(omega_shank)
    thr = np.nanmedian(wmag) + 1.5*np.nanstd(wmag)
    contact_idx = np.where(wmag > thr)[0]
    events = pd.DataFrame({
        "contact_idx": contact_idx,
        "contact_time_s": t[contact_idx] if len(contact_idx)>0 else []
    })

    # Write outputs
    out_pref = Path(args.out_prefix)
    angles = pd.DataFrame({
        "time_s": t,
        "knee_flex_deg": ang,
        "knee_frontal_proxy_deg": front
    })
    omegas = pd.DataFrame({
        "time_s": t,
        "omega_thigh_dps": omega_thigh,
        "omega_shank_dps": omega_shank,
        "omega_knee_dps": omega_knee,
    })

    ensure_parent(out_pref.with_suffix(".tmp"))
    angles.to_csv(f"{out_pref}_angles.csv", index=False)
    omegas.to_csv(f"{out_pref}_omegas.csv", index=False)
    events.to_csv(f"{out_pref}_events.csv", index=False)

    meta = {
        "calibration": parse_calib(args.calib),
        "fs": float(args.fs),
        "inputs": os.path.abspath(args.imu_csv),
        "sensors": args.sensors,
        "n_samples": int(len(t)),
    }
    with open(f"{out_pref}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
