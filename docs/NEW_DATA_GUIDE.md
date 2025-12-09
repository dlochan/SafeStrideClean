# SafeStride New Data Guide

This guide explains how to run inference-only on new trials without retraining, and how to generate clinical packs. All steps are idempotent and resume-safe. Production artifacts (Clinical v1) are not modified.

## File layout per trial

Place per-trial CSVs under a chosen folder (referred to as `IN_ROOT`). For each trial, provide:

- `<TRIAL>_imu_real.csv` (required)
- `<TRIAL>_grf.csv` (optional; enables alignment and KPI truth)
- `<TRIAL>_activity_flag.csv` (optional)

The IMU CSV must match SafeStride canonical columns for the chosen sensors.

## Run batch prediction (resume-safe)

```
python scripts/predict_batch.py \
  --in_root IN_ROOT \
  --model_pkl models/hgb_knee_dual.pkl \
  --bw_kg_default 75 \
  --window_ms 300 \
  --out_root out_newdata
```

Behavior:

- Aligns GRF to IMU if `<TRIAL>_grf.csv` is present using `scripts/auto_align_shift_grf.py`.
- Predicts GRF using frozen HGB@300 baseline via `scripts/predict_fz.py`.
- Adds uncertainty bands and builds clinical scores.
- Generates a clinical pack PDF per subject under `docs/clinical_packs/`, and copies a convenience copy to `docs/clinical_packs_new/`.
- Skips trials already processed (resume-safe).

## Outputs

- Per-trial directory under `out_newdata/<TRIAL>/` containing `predicted_fz.csv` and intermediates.
- Clinical pack PDFs under `docs/clinical_packs/` and `docs/clinical_packs_new/`.

## Notes

- Uses Matplotlib Agg; no GUI required.
- Does not modify any Clinical v1 artifacts.
