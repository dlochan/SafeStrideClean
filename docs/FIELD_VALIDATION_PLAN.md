# Field Validation Plan (Draft)

## Design

- **Split strategy**: Subject-wise GroupKFold holdouts; stratify by task.
- **Targets**:
  - Median peak Fz %BW ≤ 9 across baseline.
  - Per-subject medians ≤ 10 %BW.
  - PI coverage ≥ 0.75 per task at nominal 80%; ≥ 0.80 overall.
- **Dataset**: New field-collected IMU trials; optional GRF for alignment/calibration.

## Protocol

- **IMU placement checklist**: Sensors near knee (thigh + shank), secure mounts; record side and orientation.
- **Calibration stance**: 2–3 s quiet standing for gravity alignment.
- **Trials**: ≥ 3 per task per subject (normal, cutting, step, weighted, etc.).
- **Alignment**: If GRF present, run `scripts/auto_align_shift_grf.py` with |lag_ms| ≤ 10.

## Execution

1. Standardize new trials to canonical CSVs (`*_imu_real.csv`, optional `*_grf.csv`, `*_activity_flag.csv`).
2. Run inference-only batch (no retraining) via `scripts/predict_batch.py --in_root ...`.
3. Build clinical scores and packs.
4. If truth exists, compute KPIs and PI coverage; otherwise report QC (band width, stance time, etc.).

## Reporting

- **KPIs**: Use `tools/print_clinical_acceptance.py` and `tools/check_uncertainty_calibration.py`.
- **Evidence appendix**: Include `docs/evidence/REFERENCES.md` and provenance ledger.
- **Deliverables**: CSV summary, packs, acceptance MD/PDF.
