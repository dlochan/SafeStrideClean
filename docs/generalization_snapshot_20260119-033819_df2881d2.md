# Generalization Snapshot – 20260119-033819_df2881d2

- **run_dir**: `data/vnext_gt_real_out/vnext_fz/20260119-033819_df2881d2`
- **config**: `data/vnext_gt_real_out/tmp_overfit3d_axis3_robust_400e.yaml`
- **manifest**: `not recorded in metrics file`
- **subset filtering**: not recorded in metrics file

## Metrics (Newton space, post-denorm)

- **rmse_mean**: None
- **rmse_per_axis**:
  - Fx: None
  - Fy: None
  - Fz: None
- **corr_per_axis**:
  - Fx: None
  - Fy: None
  - Fz: None

## Units assumption

- **Units**: Newtons (post-denorm).
- **Inference**: pipeline contract (robust target_norm), presence of `target_norm.json`, and `eval_vnext` denormalization behavior.

This is descriptive only; no generalization gate yet.

## Full-manifest generalization (no subset)

- **run_dir (absolute)**: `/Volumes/Extreme SSD/safestride_clean/data/vnext_gt_real_out/vnext_fz/20260119-033819_df2881d2`
- **manifest**: `data/vnext_gt_real/manifests/vnext_val_real.csv` (from `eval_metrics_val.json`)
- **subset filtering**: none during eval (subset_indices_path removed; full manifest used)

- **rmse_mean**: ~39.85 N
- **rmse_per_axis (N)**:
  - Fx: ~22.78
  - Fy: ~24.96
  - Fz: ~71.80
- **corr_per_axis** (from `analysis_eval_full/3d_metrics_summary.json`):
  - Fx: ~0.916
  - Fy: ~0.938
  - Fz: ~0.990

- **units_detected**: `newtons`
- **num_windows (full manifest)**: 180

Artifacts:

- `<RUN_DIR>/eval_full/eval_metrics_val.json`
- `<RUN_DIR>/analysis_eval_full/3d_metrics_summary.json`
