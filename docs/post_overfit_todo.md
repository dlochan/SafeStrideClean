# Post-Overfit TODO

## 1. Frozen truths
- **Canonical run dir**: `data/vnext_gt_real_out/vnext_fz/20260119-033819_df2881d2`
- **Canonical config**: `data/vnext_gt_real_out/tmp_overfit3d_axis3_robust_400e.yaml`
- **64-window overfit contract**: Passed; 3D GRF can overfit in Newton space under the strict gate.
- **Generalization phase**: Full-validation metrics are descriptive only; no generalization gate is defined yet.
- **Training parameterization**: Robust `target_norm` + axis-weighted loss is the required 3D overfit training parameterization.

## 2. Generalization evaluation contract
- **Data**: `data/vnext_gt_real/manifests/vnext_val_real.csv` (full manifest, no subset filtering).
- **Windowing**: `window_size = 256`, `window_stride = 128`.
- **Units**: Metrics reported in **Newtons after denormalization** using `target_norm.json`.
- **Metrics to record** (per eval run):
  - `rmse_mean`
  - `rmse_per_axis["Fx"]`, `rmse_per_axis["Fy"]`, `rmse_per_axis["Fz"]`
  - `corr_per_axis["Fx"]`, `corr_per_axis["Fy"]`, `corr_per_axis["Fz"]`

## 3. Productization TODOs (in leverage order)
- **Promote canonical config** into `configs/canonical/vnext_overfit3d_canonical.yaml` (copy of `tmp_overfit3d_axis3_robust_400e.yaml`).
- **Add regression gate check script** that reads existing 3D analyzer JSON from the canonical proof run and asserts: units=newtons, Fz RMSE ≤ 150 N, Fz corr ≥ 0.90 (no training/eval).
- **Define generalization snapshot format** for full-val metrics (e.g. a small markdown template per canonical run).
- **Add CI hook (later)** to run the regression gate check and validate that analyzer/gate behavior has not regressed.

## 4. Next step
Read-only extraction of `eval/eval_metrics_val.json` from canonical run `data/vnext_gt_real_out/vnext_fz/20260119-033819_df2881d2`.
