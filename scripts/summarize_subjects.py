import argparse
import re
from pathlib import Path
import pandas as pd

SUBJ_RE = re.compile(r"^(AB\d{2})_", re.IGNORECASE)

def extract_subject(trial: str) -> str:
    m = SUBJ_RE.match(str(trial))
    return m.group(1).upper() if m else ''


def main():
    ap = argparse.ArgumentParser(description='Summarize per-subject results and best configs')
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--subject_summary', default='docs/subject_summary.csv')
    ap.add_argument('--top_configs', default='docs/top_configs_by_subject.csv')
    ap.add_argument('--shortlist_next', default='docs/shortlist_next.csv')
    args = ap.parse_args()

    out_root = Path(args.out_root)
    csv_path = out_root / 'leaderboard_all.csv'
    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()
    if df.empty:
        for p in [args.subject_summary, args.top_configs, args.shortlist_next]:
            Path(p).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_csv(p, index=False)
        print('[WARN] leaderboard_all.csv missing or empty; wrote empty summaries')
        return

    # Coerce types
    df['subject'] = df['trial'].apply(extract_subject)
    for c in ['rmse_%BW','mae_%BW']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'window_ms' in df.columns:
        df['window_ms'] = pd.to_numeric(df['window_ms'], errors='coerce').astype('Int64')

    # Per-subject summary
    grp = df.groupby('subject', dropna=False)
    summary = grp['rmse_%BW'].agg(['count','mean','median','min']).reset_index()
    summary = summary.rename(columns={'count':'n_runs','mean':'rmse_mean','median':'rmse_median','min':'rmse_best'})
    # Add best model/window for each subject
    idx = grp['rmse_%BW'].idxmin()
    best_rows = df.loc[idx.dropna().astype(int)] if len(idx) else pd.DataFrame()
    best_rows = best_rows.set_index('subject') if not best_rows.empty else pd.DataFrame()
    if not best_rows.empty:
        summary['best_model'] = summary['subject'].map(best_rows['model_kind'])
        summary['best_window'] = summary['subject'].map(best_rows['window_ms'])
    Path(args.subject_summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.subject_summary, index=False)

    # Top config per subject (single row per subject)
    cols = ['subject','trial','model_kind','window_ms','rmse_%BW','outdir']
    top_df = best_rows.reset_index()[cols] if not best_rows.empty else pd.DataFrame(columns=cols)
    top_df.to_csv(args.top_configs, index=False)

    # Shortlist for next runs (subject, model_kind, window_ms)
    shortlist_cols = ['subject','model_kind','window_ms']
    shortlist = top_df[shortlist_cols].drop_duplicates() if not top_df.empty else pd.DataFrame(columns=shortlist_cols)
    shortlist.to_csv(args.shortlist_next, index=False)

    print('[OK] wrote', args.subject_summary, args.top_configs, args.shortlist_next)

if __name__ == '__main__':
    main()
