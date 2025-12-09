import os, json, argparse, datetime as dt
import re
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

REPO_ROOT = Path(r"C:\Users\locha\Documents\safestride")
WORK_ROOT = REPO_ROOT / 'data' / 'working'
IN_CSV = REPO_ROOT / 'docs' / 'clinical_scores_summary.csv'
OUT_CSV = REPO_ROOT / 'docs' / 'clinical_scores_summary.csv'
TODO_THRESH = REPO_ROOT / 'docs' / 'evidence' / 'TODO_thresholds.csv'
SCORES = REPO_ROOT / 'docs' / 'clinical_scores_summary.csv'
OUT_DIR = REPO_ROOT / 'docs' / 'clinical_packs'

ap = argparse.ArgumentParser(description='Build clinical pack PDFs per subject')
ap.add_argument('--subject', default=None)
ap.add_argument('--leaderboard', default=None)
ap.add_argument('--outdir', default=None)
ap.add_argument('--use_grf3d', action='store_true')
ap.add_argument('--use_kin', action='store_true')
ap.add_argument('--scores_csv', default=None)
args = ap.parse_args()

df = None
scores_path = Path(args.scores_csv) if args.scores_csv else SCORES
scores_exists = scores_path.exists()
if scores_exists:
    df = pd.read_csv(scores_path)
else:
    if args.leaderboard and Path(args.leaderboard).exists():
        df = pd.read_csv(Path(args.leaderboard))
        # ensure subject derived from trial
        if 'trial' in df.columns and 'subject' not in df.columns:
            def _derive_subj(s: str) -> str:
                s = str(s)
                m = re.search(r"P\d{2}_S\d{2}", s, flags=re.IGNORECASE)
                if m: return m.group(0)[:7]
                m2 = re.search(r"P\d{2}|AB\d{2}", s, flags=re.IGNORECASE)
                return m2.group(0) if m2 else s[:4]
            df['subject'] = df['trial'].apply(_derive_subj)
    else:
        raise SystemExit(f'missing {scores_path} (and no usable --leaderboard)')
def _derive_subj(s: str) -> str:
    s = str(s)
    m = re.search(r"P\d{2}_S\d{2}", s, flags=re.IGNORECASE)
    if m:
        return m.group(0)[:7]
    m2 = re.search(r"P\d{2}", s, flags=re.IGNORECASE)
    if m2:
        return m2.group(0)[:3]
    return ''
if 'subject' not in df.columns:
    if 'trial' not in df.columns:
        raise SystemExit('clinical scores missing subject and trial')
    df['subject'] = df['trial'].apply(_derive_subj)
else:
    # fill empties from trial
    if 'trial' in df.columns:
        sub_der = df['trial'].apply(_derive_subj)
        df['subject'] = df['subject'].astype(str)
        df.loc[df['subject'].str.strip().isin(['','nan','None']), 'subject'] = sub_der

subjects = [args.subject] if args.subject else sorted(df['subject'].dropna().unique())
not_for_release = TODO_THRESH.exists()
if args.outdir:
    try:
        out_base = Path(args.outdir)
    except Exception:
        out_base = OUT_DIR
else:
    out_base = OUT_DIR
out_base.mkdir(parents=True, exist_ok=True)

for subj in subjects:
    sdf = df[df['subject']==subj].copy()
    if sdf.empty:
        continue
    out_pdf = out_base / f"{subj}_clinical_pack.pdf"
    with PdfPages(out_pdf) as pdf:
        # Page 1: Table view (top 20 rows)
        fig, ax = plt.subplots(figsize=(11,8.5))
        ax.axis('off')
        title = f'Clinical Summary — {subj}' + (' — NOT FOR RELEASE' if not_for_release else '')
        ax.set_title(title, fontsize=14)
        cols = ['trial','task','peak_fz_pctbw','loading_rate_Ns','knee_flex_rom_deg','stiff_landing','excess_load_rate','insufficient_flexion','quality_ok','pred_task']
        avail = [c for c in cols if c in sdf.columns]
        if not avail:
            avail = [c for c in ['trial','task','peak_fz_pctbw'] if c in sdf.columns]
        tab = sdf[avail].head(20).fillna('') if avail else sdf.head(20).fillna('')
        table = ax.table(cellText=tab.values, colLabels=tab.columns, loc='center')
        table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1,1.4)
        # quality and task notes
        qbad = sdf[(sdf.get('quality_ok')==False)] if 'quality_ok' in sdf.columns else pd.DataFrame()
        task_info = []
        if 'pred_task' in sdf.columns:
            task_info = sdf[['trial','pred_task']].dropna().head(5).apply(lambda r: f"{r['trial']}: {r['pred_task']}", axis=1).tolist()
        footer = []
        if not qbad.empty:
            footer.append(f"Quality flagged trials: {len(qbad)} (gray)")
        if task_info:
            footer.append("Tasks: " + "; ".join(task_info))
        if footer:
            ax.text(0.01, 0.02, " | ".join(footer), fontsize=8, transform=ax.transAxes)
        pdf.savefig(fig); plt.close(fig)
        # Page 2: Trends (plot only available columns)
        fig, axs = plt.subplots(3,1, figsize=(11,8.5))
        sdf_plot = sdf.sort_values('trial') if 'trial' in sdf.columns else sdf.copy()
        if 'loading_rate_Ns' in sdf_plot.columns:
            axs[0].plot(range(len(sdf_plot)), sdf_plot['loading_rate_Ns'], 'o-')
        axs[0].set_title('Loading rate (N/s)')
        if 'peak_fz_pctbw' in sdf_plot.columns:
            axs[1].plot(range(len(sdf_plot)), sdf_plot['peak_fz_pctbw'], 'o-')
        axs[1].set_title('Peak Fz (%BW)')
        if 'knee_flex_rom_deg' in sdf_plot.columns:
            axs[2].plot(range(len(sdf_plot)), sdf_plot['knee_flex_rom_deg'], 'o-')
        axs[2].set_title('Knee ROM (deg)')
        for a in axs: a.set_xlabel('trial index')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # Optional 3D summary page (if requested and data present)
        if args.use_grf3d:
            # find predicted_fz3d.csv for listed trials
            import numpy as np
            out_root = Path(r"E:\safestride\out_grid_multisignal\grf3d")
            rows = []
            for _, rr in sdf.iterrows():
                trial = str(rr.get('trial',''))
                p3 = out_root / trial / 'predicted_fz3d.csv'
                if not p3.exists():
                    continue
                try:
                    d = pd.read_csv(p3)
                    rec = {'trial': trial}
                    for ax in ['Fz_%BW','Fx_%BW','Fy_%BW']:
                        rec[ax+'_peak'] = float(pd.to_numeric(d.get(ax), errors='coerce').max())
                        lo = pd.to_numeric(d.get(ax+'_lo'), errors='coerce')
                        hi = pd.to_numeric(d.get(ax+'_hi'), errors='coerce')
                        if lo is not None and hi is not None and len(lo)==len(hi) and len(lo)>0:
                            rec[ax+'_band'] = float((hi - lo).mean())
                        else:
                            rec[ax+'_band'] = float('nan')
                    rows.append(rec)
                except Exception:
                    pass
            if rows:
                rdf = pd.DataFrame(rows).head(10)
                fig, axs = plt.subplots(3, 1, figsize=(11, 8.5))
                axes = [('Fz_%BW','Vertical'), ('Fx_%BW','AP'), ('Fy_%BW','ML')]
                for i, (col, name) in enumerate(axes):
                    peaks = pd.to_numeric(rdf[col+'_peak'], errors='coerce')
                    bands = pd.to_numeric(rdf[col+'_band'], errors='coerce')
                    x = np.arange(len(rdf))
                    axs[i].bar(x, peaks, yerr=bands/2.0, capsize=3)
                    axs[i].set_title(f'{name} peak %BW (error=PI band/2)')
                    axs[i].set_xticks(x)
                    axs[i].set_xticklabels(rdf['trial'], rotation=45, ha='right', fontsize=7)
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        # Longitudinal sparklines overlay (if exists)
        try:
            spark = REPO_ROOT / 'docs' / 'vNext_multisignal' / f'longitudinal_{subj}_sparklines.png'
            if spark.exists():
                fig, ax = plt.subplots(figsize=(11,2.2))
                ax.axis('off')
                img = plt.imread(str(spark))
                ax.imshow(img)
                ax.set_title('Longitudinal trends: Impulse and Loading Rate', fontsize=10)
                pdf.savefig(fig); plt.close(fig)
        except Exception:
            pass
        # Page 3: Quality notes and citations
        fig, ax = plt.subplots(figsize=(11,8.5))
        ax.axis('off')
        ax.set_title('Data Quality & Evidence Notes', fontsize=14)
        lines = []
        for _, r in sdf.iterrows():
            if not pd.isna(r.get('quality_ok')) and (not bool(r.get('quality_ok'))):
                lines.append(f"{r['trial']}: reasons={r.get('quality_reasons','')}")
        if not lines:
            lines = ['All trials passed basic quality checks.']
        # Evidence footnotes from fired flags
        cites = []
        for _, r in sdf.iterrows():
            for cat in ['stiff_landing','excess_load_rate','insufficient_flexion']:
                if bool(r.get(cat, False)):
                    c = r.get(f'{cat}_citation', None)
                    g = r.get(f'{cat}_evidence', None)
                    if pd.notna(c) and str(c):
                        cites.append(f"- {r['trial']} {cat}: citation={c} evidence={g}")
        if not cites:
            cites = ['No clinical flags fired or citations unavailable.']
        txt = 'Quality\n' + '\n'.join(lines) + '\n\nCitations\n' + '\n'.join(cites)
        ax.text(0.05, 0.9, txt, fontsize=10, va='top')
        pdf.savefig(fig); plt.close(fig)
    print('[OK] wrote', out_pdf)
