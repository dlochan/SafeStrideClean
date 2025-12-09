# Clinic Setup Guide (Field Validation)

This guide explains how to export IMU CSVs, where to drop them, and where to find results.

## Export from IMU system
- Expected schema per trial CSV: time_s, ax, ay, az, gx, gy, gz (or multi-sensor columns ax_<tag>..gz_<tag>)
- Time column must be `time_s` (seconds), monotonically increasing.
- File naming: <trial>_imu_real.csv. Optional truth: <trial>_grf.csv.

## Drop folder
- Place trial CSVs into your assigned input root, e.g. `E:\safestride\datasets\field_runs\<CLINIC_NAME>\<DATE>`.
- Each trial is a single CSV named `<trial>_imu_real.csv`.
- Optional truth file `<trial>_grf.csv` may be included for coverage analysis.

## What tasks are included
- Included: Gait, 2minWalk, FastGait, SlowGait, Running
- Excluded: Static, Synchronization, CalibrationTask, Sitting

## Running the field pipeline
- Runner: `tools/run_field_validation.py`
- Example:
  - `.\.venv\Scripts\python.exe .\tools\run_field_validation.py --in_root E:\safestride\datasets\test_field`
- Outputs:
  - Predictions under `E:\safestride\out_field\<trial>`
  - Packs under `docs/clinical_packs_field/`
  - Leaderboard CSV: `E:\safestride\out_field\leaderboard_field.csv`
  - Acceptance: `docs/field_acceptance.txt`
  - Zip: `release/SafeStride_FieldValidation_<YYYYMMDD>.zip`

## Watch-folder (optional)
- Watcher: `tools/run_watch_folder.py`
- Example:
  - `.\.venv\Scripts\python.exe .\tools\run_watch_folder.py --watch_root E:\safestride\datasets\incoming_field`
- Runs `predict_batch.py` periodically and updates packs in `docs/clinical_packs_watch/`.

## Reading packs
- Per-subject PDF packs are in `docs/clinical_packs_field/`.
- Each pack includes summary tables and plots for the subject.

## Troubleshooting
- Evidence FAIL: update `docs/evidence/dataset_registry.csv` and thresholds sources/grades; rerun.
- Zero windows: the pipeline automatically falls back 300→200→100 ms windows.
- Bad timestamps: ensure `time_s` is numeric, increasing, and sampling ~20–2000 Hz.
