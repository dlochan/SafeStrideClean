# scripts/submit_roar.py
# Render and submit Slurm jobs on ROAR for running batch_subjects_resume_dual.py slices.

from __future__ import annotations
import argparse
import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

try:
    from jinja2 import Environment, FileSystemLoader
except Exception as e:  # pragma: no cover
    Environment = None  # type: ignore
    FileSystemLoader = None  # type: ignore
    _jinja_err = e
else:
    _jinja_err = None


def _read_subjects(manifest_csv: Path, explicit_subjects: Optional[List[str]]) -> List[str]:
    if explicit_subjects:
        return explicit_subjects
    # auto-discover subjects from manifest (CSV with header 'subject')
    import csv
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    subs = sorted({r.get("subject") for r in rows if r.get("subject")})
    return [s for s in subs if s]


def _render(template_path: Path, context: Dict) -> str:
    if Environment is None:
        raise SystemExit(f"Jinja2 not available: {_jinja_err}\nInstall with `pip install jinja2`." )
    env = Environment(loader=FileSystemLoader(str(template_path.parent)))
    tpl = env.get_template(template_path.name)
    return tpl.render(**context)


def _submit(script_path: Path, dependency: Optional[int] = None, dry_run: bool = False) -> Optional[int]:
    cmd = ["sbatch"]
    if dependency is not None:
        cmd += [f"--dependency=afterok:{dependency}"]
    cmd += [str(script_path)]
    print("\n>", " ".join(cmd))
    if dry_run:
        return None
    cp = subprocess.run(cmd, capture_output=True, text=True)
    out = (cp.stdout or "") + "\n" + (cp.stderr or "")
    m = re.search(r"Submitted batch job (\d+)", out)
    if cp.returncode != 0 or not m:
        print("[WARN] sbatch returned:", out)
        return None
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser(description="Submit ROAR Slurm jobs for safestride batches.")
    ap.add_argument("--template", default="slurm/safestride.sbatch.j2")
    ap.add_argument("--manifest_csv", required=True)
    ap.add_argument("--shortlist_csv", required=True)
    ap.add_argument("--subjects", nargs="*", default=None, help="If omitted, infer from manifest")
    ap.add_argument("--project_root", required=True)
    ap.add_argument("--log_dir", default="slurm/logs")
    ap.add_argument("--sensor_set", default="lpthigh,lshank")
    ap.add_argument("--work_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--fs", type=int, default=200)
    ap.add_argument("--default_bw_kg", type=float, default=78.9)
    # Slurm resources
    ap.add_argument("--time", default="02:00:00")
    ap.add_argument("--cpus", type=int, default=4)
    ap.add_argument("--mem", default="16G")
    ap.add_argument("--partition", default=None)
    ap.add_argument("--account", default=None)
    ap.add_argument("--gres", default=None)
    # Behavior
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry_run", action="store_true", help="Do not call sbatch; just print")
    ap.add_argument("--chain", action="store_true", help="Submit jobs with afterok dependency chaining")
    args = ap.parse_args()

    template_path = Path(args.template)
    subjects = _read_subjects(Path(args.manifest_csv), args.subjects)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    render_dir = Path("slurm/rendered")
    render_dir.mkdir(parents=True, exist_ok=True)

    last_job_id: Optional[int] = None
    for subj in subjects:
        context = {
            "job_name": f"safestride_{subj}",
            "log_dir": os.path.abspath(args.log_dir),
            "time": args.time,
            "cpus": args.cpus,
            "mem": args.mem,
            "partition": args.partition,
            "account": args.account,
            "gres": args.gres,
            "project_root": args.project_root,
            "manifest_csv": os.path.abspath(args.manifest_csv),
            "shortlist_csv": os.path.abspath(args.shortlist_csv),
            "subjects": [subj],
            "sensor_set": args.sensor_set,
            "work_root": args.work_root,
            "out_root": args.out_root,
            "fs": int(args.fs),
            "overwrite": bool(args.overwrite),
            "dry_run": False,  # inside batch we want real run by default
            "default_bw_kg": float(args.default_bw_kg),
        }
        script_text = _render(template_path, context)
        script_path = render_dir / f"safestride_{subj}.sbatch"
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_text)
        dep = last_job_id if args.chain else None
        job_id = _submit(script_path, dependency=dep, dry_run=args.dry_run)
        if job_id:
            print(f"[JOB] {subj} -> {job_id}")
        last_job_id = job_id or last_job_id


if __name__ == "__main__":
    main()
