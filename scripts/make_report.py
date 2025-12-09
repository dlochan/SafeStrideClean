import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def safe_read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def add_text_table(ax, df: pd.DataFrame, title: str, n=20):
    ax.axis('off')
    ax.set_title(title)
    if df.empty:
        ax.text(0.5,0.5,'No data', ha='center', va='center')
        return
    head = df.head(n).copy()
    # Limit columns
    cols = [c for c in ['trial','sensortag','model_kind','window_ms','rmse_%BW','mae_%BW'] if c in head.columns]
    head = head[cols]
    txt = head.to_string(index=False)
    ax.text(0.01, 0.99, txt, va='top', family='monospace')


def add_image_page(pdf: PdfPages, img_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(11,8.5))
    ax.axis('off')
    ax.set_title(title)
    try:
        import matplotlib.image as mpimg
        img = mpimg.imread(str(img_path))
        ax.imshow(img)
    except Exception as e:
        ax.text(0.5,0.5,f'Failed to load {img_path}: {e}', ha='center', va='center')
    pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Create Windows batch summary PDF')
    ap.add_argument('--out_root', default=r'E:\safestride\out_grid')
    ap.add_argument('--fig_dir', default='docs/figures')
    ap.add_argument('--outfile', default='docs/Windows_Batch_Summary.pdf')
    args = ap.parse_args()

    out_root = Path(args.out_root)
    csv_path = out_root / 'leaderboard_all.csv'
    df = safe_read_csv(csv_path)

    # Ensure figures exist or generate placeholders
    from pathlib import Path as P
    fig_dir = P(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)
    figs = {
        'rmse_bw_by_model_window.png': fig_dir / 'rmse_bw_by_model_window.png',
        'rmse_bw_hist.png': fig_dir / 'rmse_bw_hist.png',
        'top10_runs.png': fig_dir / 'top10_runs.png',
    }
    for name, path in figs.items():
        if not path.exists():
            plt.figure(figsize=(8,4)); plt.text(0.5,0.5,f'Missing {name}', ha='center', va='center'); plt.axis('off'); plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()

    out_pdf = Path(args.outfile)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        # Title page
        fig, ax = plt.subplots(figsize=(11,8.5))
        ax.axis('off')
        ax.text(0.5, 0.7, 'SafeStride Windows Batch Summary', ha='center', va='center', fontsize=18)
        ax.text(0.5, 0.5, f'Out root: {out_root}', ha='center', va='center')
        ax.text(0.5, 0.45, f'Rows in leaderboard: {len(df)}', ha='center', va='center')
        pdf.savefig(fig); plt.close(fig)

        # Image pages
        add_image_page(pdf, figs['rmse_bw_by_model_window.png'], 'RMSE by model × window')
        add_image_page(pdf, figs['rmse_bw_hist.png'], 'RMSE histogram')
        add_image_page(pdf, figs['top10_runs.png'], 'Top 10 runs')

        # Table page
        fig, ax = plt.subplots(figsize=(11,8.5))
        add_text_table(ax, df.sort_values(by='rmse_%BW', ascending=True, na_position='last'), 'Top 20 leaderboard rows')
        pdf.savefig(fig); plt.close(fig)

    print('[OK] wrote', out_pdf)


if __name__ == '__main__':
    main()
