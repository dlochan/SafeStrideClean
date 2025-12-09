import argparse, csv, os, shutil, subprocess
from pathlib import Path
import json

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = REPO_ROOT / '.venv' / 'Scripts' / 'python.exe'
try:
    from tools import path_config as PC  # type: ignore
except Exception:
    PC = None

ap = argparse.ArgumentParser(description='Batch wrapper: call predict_fz.py for each row in a CSV manifest or scan an input root of new trials')
ap.add_argument('--manifest_csv', help='CSV with columns: imu_csv, model_pkl, bw_kg, window_ms, trial, task, outdir')
ap.add_argument('--in_root', help='Root folder containing per-trial CSVs: *_imu_real.csv, optional *_grf.csv, optional *_activity_flag.csv')
ap.add_argument('--model_pkl', help='Path to frozen baseline model (required with --in_root)')
ap.add_argument('--bw_kg_default', type=float, default=None, help='Default BW kg if not per-trial meta present')
ap.add_argument('--window_ms', type=int, default=300)
ap.add_argument('--out_root', type=str, default=(str(PC.OUT_ROOT/'out_newdata') if PC is not None else str(REPO_ROOT/'out_newdata')))
args = ap.parse_args()

def run_manifest(csv_path: Path):
    with open(csv_path, newline='') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            outdir = Path(r['outdir']); outdir.mkdir(parents=True, exist_ok=True)
            # resume-safe: skip if predicted and pack present
            pack_pdf = REPO_ROOT/'docs'/'clinical_packs'/f"{r['trial'][:4]}_clinical_pack.pdf"
            if (outdir/'predicted_fz.csv').exists() and pack_pdf.exists():
                continue
            cmd = [str(PY), str(REPO_ROOT/'scripts'/'predict_fz.py'),
                   '--imu_csv', r['imu_csv'],
                   '--model_pkl', r['model_pkl'],
                   '--bw_kg', str(r['bw_kg']),
                   '--window_ms', str(r.get('window_ms', args.window_ms)),
                   '--trial', r['trial'],
                   '--task', r['task'],
                   '--outdir', r['outdir']]
            subprocess.run(cmd, check=True)

def _auto_discover_model() -> str | None:
    # Prefer frozen models in release/models matching hgb w300 kneepair
    rel = REPO_ROOT/'release'/'models'
    if rel.exists():
        for p in rel.rglob('*.pkl'):
            s = str(p).lower()
            if 'hgb' in s and 'w300' in s and 'kneepair' in s:
                return str(p)
    # Fallback to out_grid AB01 HGB w300
    og = Path(r"E:\safestride\out_grid")
    if og.exists():
        for p in og.rglob('model.pkl'):
            s = str(p).lower()
            if 'ab01' in s and 'hgb' in s and 'w300' in s:
                return str(p)
    return None

def _get_mass_for_subject(subj: str) -> float:
    try:
        out = subprocess.run([str(PY), str(REPO_ROOT/'tools'/'get_subject_mass.py'), '--subject', subj], capture_output=True, text=True, check=False)
        v = float(out.stdout.strip()) if out.stdout else float('nan')
        if v and v>0: return v
    except Exception:
        pass
    return 75.0

def run_inroot(in_root: Path):
    model = args.model_pkl or _auto_discover_model()
    if not model:
        raise SystemExit('No frozen model found (need HGB@300); aborting')
    out_root = Path(args.out_root); out_root.mkdir(parents=True, exist_ok=True)
    paths = None
    if PC is not None:
        try:
            from tools.path_config import SafeStridePaths as _SSP  # type: ignore
            paths = _SSP.from_env_or_config()
        except Exception:
            paths = None
    if paths is not None:
        doc_root = paths.doc_root
    else:
        doc_root = (PC.DOC_ROOT if PC is not None else (REPO_ROOT/'docs'))
    packs_dir = (doc_root/'clinical_packs_grouvel'); packs_dir.mkdir(parents=True, exist_ok=True)
    skips_csv = (doc_root/'grouvel_sandbox_skips.csv')
    # trial-name filter
    include_kw = ['gait','2minwalk','fastgait','slowgait','running']
    exclude_kw = ['static','synchronization','calibrationtask','sitting']
    # scan for trials: accept *_imu_real.csv, *_imu.csv, and .csv.gz variants
    def iter_imu_files(root: Path):
        pats = ['*_imu_real.csv', '*_imu.csv', '*_imu_real.csv.gz', '*_imu.csv.gz', '*.csv', '*.csv.gz']
        seen = set()
        for pat in pats:
            for p in root.rglob(pat):
                sp = str(p)
                if sp in seen:
                    continue
                seen.add(sp)
                # skip likely GRF files
                if p.name.lower().endswith('_grf.csv') or p.name.lower().endswith('_grf.csv.gz'):
                    continue
                yield p
    def trial_from_name(p: Path) -> str | None:
        name = p.name
        for suf in ['_imu_real.csv', '_imu.csv', '_imu_real.csv.gz', '_imu.csv.gz']:
            if name.endswith(suf):
                return name[: -len(suf)]
        # fallback: generic trial.csv or trial.csv.gz
        if name.endswith('.csv'):
            return name[:-4]
        if name.endswith('.csv.gz'):
            return name[:-7]
        return None
    # scan for trials
    for imu_csv in iter_imu_files(in_root):
        trial = trial_from_name(imu_csv)
        if not trial:
            continue
        name_l = trial.lower()
        inc = any(k in name_l for k in include_kw)
        exc = any(k in name_l for k in exclude_kw)
        if not inc or exc:
            try:
                with open(skips_csv, 'a', encoding='utf-8', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=['trial','reason'])
                    if f.tell()==0: w.writeheader()
                    w.writerow({'trial': trial, 'reason': 'task_filter'})
            except Exception:
                pass
            continue
        # derive subject from trial using regex: prefer Pnn_Snn else Pnn
        import re as _re
        subj = None
        m = _re.search(r"P\d{2}_S\d{2}", trial, flags=_re.IGNORECASE)
        if m:
            subj = m.group(0)[:7]
        else:
            m2 = _re.search(r"P\d{2}", trial, flags=_re.IGNORECASE)
            subj = m2.group(0)[:3] if m2 else trial[:4]
        task = 'unknown'
        grf_csv = imu_csv.with_name(trial + '_grf.csv')
        outdir = out_root/ trial
        outdir.mkdir(parents=True, exist_ok=True)
        pack_pdf = REPO_ROOT/'docs'/'clinical_packs'/f"{subj}_clinical_pack.pdf"
        if (outdir/'predicted_fz.csv').exists() and pack_pdf.exists():
            continue
        # if GRF present, align (new args)
        if grf_csv.exists():
            try:
                subprocess.run([str(PY), str(REPO_ROOT/'scripts'/'auto_align_shift_grf.py'),
                                '--imu', str(imu_csv), '--grf_in', str(grf_csv),
                                '--grf_out', str((REPO_ROOT/'data'/'working'/f"{trial}_grf_active_shifted.csv"))], check=False)
            except Exception:
                pass
        # Subject body mass
        bw = args.bw_kg_default if args.bw_kg_default else _get_mass_for_subject(subj)
        # adaptive windows
        for w in (300, 200, 100):
            try:
                subprocess.run([str(PY), str(REPO_ROOT/'scripts'/'predict_fz.py'),
                                '--imu_csv', str(imu_csv), '--model_pkl', str(model),
                                '--bw_kg', str(bw), '--window_ms', str(w),
                                '--trial', trial, '--task', task, '--outdir', str(outdir)], check=True)
            except subprocess.CalledProcessError:
                pass
            pred = outdir/'predicted_fz.csv'
            if pred.exists():
                try:
                    # require at least 5 rows
                    import pandas as pd
                    if len(pd.read_csv(pred)) >= 5:
                        break
                    else:
                        pred.unlink(missing_ok=True)
                except Exception:
                    break
        else:
            # none succeeded
            try:
                with open(skips_csv, 'a', encoding='utf-8', newline='') as f:
                    w = csv.DictWriter(f, fieldnames=['trial','reason'])
                    if f.tell()==0: w.writeheader()
                    w.writerow({'trial': trial, 'reason': 'short_sequence'})
            except Exception:
                pass
            continue
        # add uncertainty (idempotent) and build pack for subject and copy
        try:
            subprocess.run([str(PY), str(REPO_ROOT/'tools'/'add_uncertainty.py'), '--trial', trial, '--outdir', str(outdir), '--default_rmse_bw', '7.0'], check=False)
            subprocess.run([str(PY), str(REPO_ROOT/'scripts'/'make_clinical_pack.py'), '--subject', subj], check=False)
            src_pack = REPO_ROOT/'docs'/'clinical_packs'/f"{subj}_clinical_pack.pdf"
            if src_pack.exists():
                shutil.copy2(src_pack, packs_dir/(src_pack.name))
        except Exception:
            pass

if args.manifest_csv:
    run_manifest(Path(args.manifest_csv))
elif args.in_root:
    run_inroot(Path(args.in_root))
else:
    raise SystemExit('Provide either --manifest_csv or --in_root')

print('[OK] predict_batch completed')
