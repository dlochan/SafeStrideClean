# MP_CONVERGE_3D – Canonical Entry Point

## Overfit contract (64-window subset)

- **Canonical run_dir**: `data/vnext_gt_real_out/vnext_fz/20260119-033819_df2881d2`
- **Subset**: 64 fixed windows drawn from the canonical validation manifest.
- **Windowing**: `window_size = 256`, `window_stride = 128`.
- **Units**: 3D GRF in **Newtons** (post-denorm via `target_norm.json`).
- **Canonical PASS numbers** (from `analysis_eval/3d_metrics_summary.json`):
  - `num_windows = 64`
  - `window_len = 256`
  - `Fz_rmse ≈ 63.46 N`
  - `Fz_corr ≈ 0.9931`
  - `gate.status = "PASS"`

The overfit contract asserts that these conditions hold:

- `units_detected ∈ {"newtons","newton","n"}`
- `num_windows == 64`
- `window_len == 256`
- `axis_summaries["Fz"]["rmse"] ≤ 150.0`
- `axis_summaries["Fz"]["corr"] ≥ 0.90`
- `gate.status == "PASS"`

## One-liner: smoke (overfit + generalization)

Runs the canonical overfit contract and prints a one-screen summary for both the 64‑window subset and the full-manifest evaluation:

```bash
bash scripts/mp_converge_3d_smoke.sh
```

This script is read-only and uses existing artifacts under the canonical `run_dir`.

## One-liner: contract-only check (CI-safe)

Runs only the 64-window overfit contract on the canonical analyzer JSON:

```bash
python3 scripts/check_overfit3d_contract.py
```

This is what `scripts/ci_check_overfit3d_contract.sh` wraps for CI.

## One-liner: full-manifest eval (when desired)

To re-run full-manifest evaluation **for the canonical run only** (no training):

```bash
bash scripts/run_generalization_eval.sh data/vnext_gt_real_out/vnext_fz/20260119-033819_df2881d2 --device mps
```

This assumes the canonical run directory already exists with checkpoints and normalization stats.

### Full-manifest outputs

For a given `RUN_DIR` (e.g. `data/vnext_gt_real_out/vnext_fz/20260119-033819_df2881d2`), the full-manifest helper writes:

- `<RUN_DIR>/eval_full/eval_metrics_val.json`
- `<RUN_DIR>/analysis_eval_full/3d_metrics_summary.json`

These contain the full 3D RMSE and correlation metrics (in Newtons) for ~180 windows on the canonical validation manifest.
