# scripts/standardize_gt_trial_active.py
import argparse, sys, os, datetime
from pathlib import Path
import pandas as pd

from src.adapters.gt_noncyclic import (
    load_gt_imu_real,
    load_gt_imu_multi,
    load_gt_grf_total,
    load_gt_activity_flag,
    filter_active,
)

def _setup_logging(default_name: str, user_log: str | None) -> None:
    log_path = None
    if user_log:
        log_path = user_log
    else:
        logs_root = os.getenv("SAFESTRIDE_LOGS_ROOT", r"C:\\Users\\locha\\Documents\\safestride\\logs")
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = str(Path(logs_root) / f"{default_name}_{ts}.log")
    try:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        f = open(log_path, "a", encoding="utf-8")
    except Exception:
        return
    class _Tee:
        def __init__(self, fobj):
            self.f = fobj
            self._stdout = sys.stdout
            self._stderr = sys.stderr
        def write(self, s):
            try:
                self.f.write(s)
            except Exception:
                pass
            return self._stdout.write(s)
        def flush(self):
            try:
                self.f.flush()
            except Exception:
                pass
            return self._stdout.flush()
    tee = _Tee(f)
    sys.stdout = tee
    sys.stderr = tee


def main():
    ap = argparse.ArgumentParser(description="Standardize GT trial; filter to active frames; support multi-sensor.")
    ap.add_argument("--imu_in", required=True)
    ap.add_argument("--grf_in", required=True)
    ap.add_argument("--flag_in", required=True)
    # Either use legacy --sensor (single) OR --sensors (multi, comma-separated)
    ap.add_argument("--sensor", default=None, help="Single sensor (legacy). e.g., rshank")
    ap.add_argument("--sensors", default=None, help="Comma-separated sensors for multi-sensor. e.g., lpthigh,lshank")
    ap.add_argument("--imu_out", required=True)
    ap.add_argument("--grf_out", required=True)
    ap.add_argument("--skip_on_error", action="store_true", help="If set, print a warning and exit 0 on recoverable errors.")
    ap.add_argument("--log_file", default=None, help="Optional log file to tee output; otherwise logs to LOGS_ROOT")
    args = ap.parse_args()

    try:
        # Setup logging
        stem = Path(args.imu_in).stem if args.imu_in else "standardize"
        _setup_logging(f"standardize_{stem}", args.log_file)
        # 1) IMU (single or multi)
        if args.sensors:
            sensors = [s.strip() for s in args.sensors.split(",") if s.strip()]
            if len(sensors) == 0:
                raise ValueError("--sensors provided but empty.")
            imu_df = load_gt_imu_multi(args.imu_in, sensors=sensors)
            # If specific dual-set requested, rename to thigh/shank schema
            # Expected sensors order: ["lpthigh","lshank"] -> rename columns to ax_thigh, ..., gz_shank
            alias = {"lpthigh": "thigh", "lshank": "shank"}
            if len(sensors) == 2 and sensors[0] in alias and sensors[1] in alias:
                new_cols = {}
                for s in sensors:
                    suf = alias[s]
                    for comp in ("ax","ay","az","gx","gy","gz"):
                        old = f"{comp}_{s}"
                        if old in imu_df.columns:
                            new_cols[old] = f"{comp}_{suf}"
                imu_df = imu_df.rename(columns=new_cols)
        else:
            if not args.sensor:
                raise ValueError("Provide either --sensor (single) OR --sensors (comma-separated).")
            imu_df = load_gt_imu_real(args.imu_in, sensor=args.sensor)

        # 2) GRF
        grf_df = load_gt_grf_total(args.grf_in)

        # 3) Activity flag & filter
        flag_df = load_gt_activity_flag(args.flag_in)
        imu_act, grf_act = filter_active(imu_df, grf_df, flag_df)

        # Edge case: no active frames
        if len(imu_act) == 0 or len(grf_act) == 0:
            raise ValueError("No active frames after filtering.")

        # 4) Save and write _SUCCESS marker
        Path(args.imu_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.grf_out).parent.mkdir(parents=True, exist_ok=True)
        imu_act.to_csv(args.imu_out, index=False)
        grf_act.to_csv(args.grf_out, index=False)
        # marker in common parent
        success_path = Path(args.imu_out).parent / "_SUCCESS"
        try:
            success_path.write_text("ok\n", encoding="utf-8")
        except Exception:
            pass

        print("[OK] wrote:")
        print(f"  {os.path.abspath(args.imu_out)}")
        print(f"  {os.path.abspath(args.grf_out)}")
        print("IMU columns:", list(imu_act.columns)[:12], "...")
        print("GRF columns:", list(grf_act.columns))

    except Exception as e:
        msg = f"[SKIP] {e.__class__.__name__}: {e}"
        if args.skip_on_error:
            print(msg)
            sys.exit(0)  # successful skip
        else:
            print(msg, file=sys.stderr)
            sys.exit(1)

if __name__ == "__main__":
    main()
