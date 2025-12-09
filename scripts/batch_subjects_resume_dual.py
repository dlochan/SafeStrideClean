# scripts/batch_subjects_resume_dual.py
# Resume-safe batch runner for local machine. Supports dual-sensor standardization and feature pipeline.

from __future__ import annotations
import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess

from src.config import WORK_ROOT as CFG_WORK_ROOT, OUT_ROOT as CFG_OUT_ROOT, FS_HZ as CFG_FS_HZ, BW_BY_SUBJECT

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
FAILED_LOG = LOGS_DIR / "failed_trials.csv"


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


@dataclass
class CLI:
    manifest_csv: Path
    shortlist_csv: Path
    subjects: Optional[List[str]]
    sensor_set_fallback: str
    work_root: Path
    out_root: Path
    fs_hz: int
    resume: bool
    overwrite: bool
    dry_run: bool
    default_bw_kg: Optional[float]


def sh(cmd: List[str], dry_run: bool = False) -> int:
    print("\n>", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    cp = subprocess.run(cmd)
    return cp.returncode


def run_is_done(outdir: Path) -> bool:
    pred = outdir / "predicted_fz.csv"
    met = outdir / "eval" / "metrics_eval.json"
    return pred.is_file() and met.is_file()


def write_failed(row: Dict[str, str], step: str, exc: Exception | str):
    FAILED_LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not FAILED_LOG.exists()
    with open(FAILED_LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["subject","trial","sensors","model","window_ms","step","error"]) 
        w.writerow([
            row.get("subject",""),
            row.get("trial",""),
            row.get("sensors",""),
            row.get("model",""),
            row.get("window_ms",""),
            step,
            str(exc),
        ])


def _pick_subjects(manifest_rows: List[Dict[str,str]], subjects_cli: Optional[List[str]]) -> List[str]:
    all_subjects = sorted(set(r["subject"] for r in manifest_rows if "subject" in r))
    if subjects_cli:
        want = set(subjects_cli)
        return [s for s in all_subjects if s in want]
    return all_subjects


def _get_bw_kg(subj: str, default_bw_kg: Optional[float]) -> Optional[float]:
    return BW_BY_SUBJECT.get(subj, default_bw_kg)


def _sensor_key(sensors: str) -> str:
    # sanitize for filesystem
    return sensors.replace(",", "").replace(" ", "")


def plan_and_run(cli: CLI) -> Dict[str,int]:
    manifest = _read_csv(cli.manifest_csv)
    picks = _read_csv(cli.shortlist_csv)

    # Build index: task key -> shortlist rows
    # Expect shortlist["trial"] like "AB01_<task>"; derive <task> by removing leading subject
    def shortlist_for_task(task: str) -> List[Dict[str,str]]:
        ab01_key = f"AB01_{task}"
        return [r for r in picks if r.get("trial") == ab01_key]

    subjects = _pick_subjects(manifest, cli.subjects)
    print(f"Subjects to run: {subjects}")

    # Summary counters
    n_trials = 0
    scheduled = 0
    skipped_done = 0
    failed = 0

    for subj in subjects:
        sub_rows = [r for r in manifest if r.get("subject") == subj]
        print(f"\n=== Subject {subj} | {len(sub_rows)} entries ===")
        bw_kg = _get_bw_kg(subj, cli.default_bw_kg)
        if bw_kg is None:
            print(f"[WARN] No bodyweight for {subj}; skipping subject.")
            continue

        for r in sub_rows:
            # manifest expected columns: subject, trial, imu_in, grf_in, flag_in
            trial = r.get("trial")
            if not trial:
                continue
            # Robustly handle trial names that may be either task-only (e.g., cutting_leftfast)
            # or already prefixed with subject (e.g., AB02_cutting_leftfast)
            if isinstance(trial, str) and trial.startswith(f"{subj}_"):
                trial_name = trial
                task_key = trial.split("_", 1)[1]
            else:
                task_key = trial
                trial_name = f"{subj}_{trial}"

            imu_in = r.get("imu_in") or r.get("imu_real")
            grf_in = r.get("grf_in") or r.get("grf")
            flag_in= r.get("flag_in") or r.get("activity_flag")
            if not (imu_in and grf_in and flag_in):
                write_failed({**r, "sensors":"-"}, "input_missing", "manifest missing paths")
                failed += 1
                continue

            # Pre-check: if inputs missing, skip and log
            if not (Path(imu_in).is_file() and Path(grf_in).is_file() and Path(flag_in).is_file()):
                write_failed({**r, "sensors":"-"}, "input_missing", "one or more input files not found")
                print(f"[SKIP] inputs not found for {trial_name}")
                failed += 1
                continue

            # Shortlist rows for this task
            S = shortlist_for_task(task_key)
            if not S:
                print(f"[WARN] no shortlist for AB01_{task_key}; using fallback sensor_set={cli.sensor_set_fallback} with rf w300")
                S = [{"sensor": cli.sensor_set_fallback, "model": "rf", "window_ms": "300"}]

            # Standardized working files
            imu_work = cli.work_root / f"{trial_name}_imu_active.csv"
            grf_work = cli.work_root / f"{trial_name}_grf_active.csv"
            grf_shift= cli.work_root / f"{trial_name}_grf_active_shifted.csv"

            # Determine sensorset for standardize: prefer first shortlist row's sensor if comma-separated, else fallback
            s0 = S[0].get("sensor", "")
            sensorset = s0 if ("," in s0) else cli.sensor_set_fallback

            # 1) Standardize (resume on _SUCCESS and files present)
            success_marker = imu_work.parent / "_SUCCESS"
            std_cached = imu_work.is_file() and grf_work.is_file() and success_marker.is_file()
            if cli.resume and std_cached and not cli.overwrite:
                print(f"[RESUME] Cached standardized files for {trial_name}")
            else:
                cmd = [
                    sys.executable, "scripts/standardize_gt_trial_active.py",
                    "--imu_in", imu_in,
                    "--grf_in", grf_in,
                    "--flag_in", flag_in,
                    "--sensors", sensorset,
                    "--imu_out", str(imu_work),
                    "--grf_out", str(grf_work),
                ]
                rc = sh(cmd, cli.dry_run)
                if rc != 0:
                    write_failed({**r, "sensors": sensorset}, "standardize", f"rc={rc}")
                    failed += 1
                    continue

            # 2) Align (resume if shifted exists)
            if cli.resume and grf_shift.is_file() and not cli.overwrite:
                print(f"[RESUME] Cached aligned GRF for {trial_name}")
            else:
                cmd = [
                    sys.executable, "scripts/auto_align_shift_grf.py",
                    "--imu", str(imu_work),
                    "--grf_in", str(grf_work),
                    "--grf_out", str(grf_shift),
                    "--fs", str(cli.fs_hz),
                ]
                rc = sh(cmd, cli.dry_run)
                if rc != 0:
                    write_failed({**r, "sensors": sensorset}, "align", f"rc={rc}")
                    failed += 1
                    continue

            # Loop shortlist configurations
            for cfg in S:
                try:
                    model = str(cfg.get("model", "rf"))
                    w = int(cfg.get("window_ms", 300))
                    sensors = str(cfg.get("sensor", sensorset))
                    # Build features if not present
                    prefix = cli.work_root / f"{trial_name}_dual_w{w}"
                    x_path = prefix.with_name(prefix.name + "_X").with_suffix(".parquet")
                    y_path = prefix.with_name(prefix.name + "_y").with_suffix(".parquet")
                    meta_p = prefix.with_name(prefix.name + "_meta").with_suffix(".json")

                    if not (x_path.is_file() and y_path.is_file()) or cli.overwrite:
                        cmd = [
                            sys.executable, "scripts/make_features.py",
                            "--imu_csv", str(imu_work),
                            "--grf_csv", str(grf_shift),
                            "--window_ms", str(w),
                            "--out_prefix", str(prefix),
                            "--fs", str(cli.fs_hz),
                        ]
                        rc = sh(cmd, cli.dry_run)
                        if rc != 0:
                            write_failed({**r, "sensors": sensors, "model": model, "window_ms": str(w)}, "make_features", f"rc={rc}")
                            failed += 1
                            continue

                    # Train outdir name
                    sensor_key = _sensor_key(sensors)
                    outdir = cli.out_root / f"{trial_name}_{sensor_key}_{model}_w{w}"

                    # Resume if done
                    if cli.resume and not cli.overwrite and run_is_done(outdir):
                        print(f"[RESUME] Skipping already complete: {outdir}")
                        skipped_done += 1
                        continue

                    outdir.mkdir(parents=True, exist_ok=True)
                    # Train
                    cmd = [
                        sys.executable, "-m", "src.train",
                        "--features_X", str(x_path),
                        "--features_y", str(y_path),
                        "--bw_kg", str(bw_kg),
                        "--model_kind", model,
                        "--outdir", str(outdir),
                    ]
                    rc = sh(cmd, cli.dry_run)
                    if rc != 0:
                        write_failed({**r, "sensors": sensors, "model": model, "window_ms": str(w)}, "train", f"rc={rc}")
                        failed += 1
                        continue

                    # Eval
                    cmd = [
                        sys.executable, "-m", "src.eval_compare",
                        "--true_grf_csv", str(grf_shift),
                        "--pred_grf_csv", str(outdir/"predicted_fz.csv"),
                        "--outdir", str(outdir/"eval"),
                        "--bw_kg", str(bw_kg),
                    ]
                    rc = sh(cmd, cli.dry_run)
                    if rc != 0:
                        write_failed({**r, "sensors": sensors, "model": model, "window_ms": str(w)}, "eval", f"rc={rc}")
                        failed += 1
                        continue

                    scheduled += 1
                except (OSError, ValueError) as e:
                    write_failed({**r, "sensors": sensors, "model": model, "window_ms": str(w)}, "exception", e)
                    failed += 1
                    continue

            n_trials += 1

    print(f"\n[SUMMARY] trials={n_trials} scheduled={scheduled} skipped_done={skipped_done} failed={failed}")
    return {"trials": n_trials, "scheduled": scheduled, "skipped_done": skipped_done, "failed": failed}


def main():
    ap = argparse.ArgumentParser(description="Resume-safe dual-sensor batch runner with dry-run.")
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--shortlist_csv", required=True, help="Shortlist CSV from AB01 with columns: trial,sensor,model,window_ms")
    ap.add_argument("--subjects", nargs="*", default=None, help="Optional list of subjects to run. Default: all in manifest")
    ap.add_argument("--sensor_set", default="lpthigh,lshank", help="Fallback dual-sensor set if shortlist row sensor is not multi (default: lpthigh,lshank)")
    ap.add_argument("--work_root", default=str(CFG_WORK_ROOT))
    ap.add_argument("--out_root",  default=str(CFG_OUT_ROOT))
    ap.add_argument("--fs", type=int, default=int(CFG_FS_HZ))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    ap.add_argument("--default_bw_kg", type=float, default=None, help="Override BW if subject not found in config map")
    args = ap.parse_args()

    cli = CLI(
        manifest_csv=Path(args.manifest_csv),
        shortlist_csv=Path(args.shortlist_csv),
        subjects=args.subjects if args.subjects else None,
        sensor_set_fallback=args.sensor_set,
        work_root=Path(args.work_root),
        out_root=Path(args.out_root),
        fs_hz=int(args.fs),
        resume=bool(args.resume),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
        default_bw_kg=args.default_bw_kg,
    )

    plan_and_run(cli)


if __name__ == "__main__":
    main()
