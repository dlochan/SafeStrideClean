# SafeStride – Project Guide for New Contributors

This README is a newcomer-friendly orientation to the SafeStride codebase. It explains what the project does, how it’s organized, how paths and data are managed (E: drive heavy roots), and the most common workflows to run, debug, and extend the system. It is safe to read top-to-bottom and follow along.

---

## What SafeStride Does

- Ingests IMU CSVs and normalizes column names to a canonical schema.
- Predicts vertical Ground Reaction Force (vGRF) time-series from IMUs using frozen baseline models.
- Computes MVP metrics and flags (e.g., high impact, stiff landing, asymmetry).
- Evaluates rule-based Risk Engine v1 with evidence-backed thresholds.
- Produces resume-safe outputs, logs, and validation reports.

This repo keeps source code on C:, while heavy artifacts live on E:. Determinism and auditability are core values: no retraining during standard runs, and evidence enforcement gates are enabled.

---

## Architecture at a Glance

- Canonical IMU schema: `time_s`, and per-sensor axes `ax`, `ay`, `az` (required), optionally `gx`, `gy`, `gz`.
- Multi-sensor support: normalized names can be suffixed, e.g., `ax_lshank`.
- Centralized paths: `tools/path_config.py` resolves heavy roots on E: and ensures directories exist.
- Evidence enforcement: thresholds carry sources/grades; `tools/evidence_registry.py --enforce` validates documentation before runs.
- Resume-safety: all runners and tools avoid redoing work and can be resumed.

---

## Repository Layout (key folders)

- `configs/`
  - `dataset.yaml` – Declares dataset root and E: heavy roots (data, work, out, release, logs, docs). Also default `fs_hz` and helper meta.
- `docs/`
  - `validation/` – Summaries and validation report outputs.
  - `risk_engine/` – Risk Engine reports and features.
  - `PATHS_CURRENT.md` – Auto-generated current path resolution.
  - `README_PROJECT.md` – This newcomer guide.
- `scripts/`
  - `run_internal_validation.py` – End-to-end internal validation (MVP + Risk) with resume-safety and E: roots.
  - `run_safestride_mvp.py` – MVP runner over new trials (optional kinematics + risk).
  - `predict_batch.py` – Batch wrapper for predictions from a manifest or scanning a folder.
- `src/`
  - Core features and data loading helpers.
- `tools/`
  - `path_config.py` – Centralized path resolution (prefers E: roots).
  - `normalize_imu_schema.py` – Deterministic IMU schema normalization and logging.
  - `inventory_imu_headers.py` – Builds a CSV inventory of IMU headers.
  - `build_kinematic_surrogates.py` – Derives kinematic surrogate features from IMU and vGRF predictions.
  - `risk_engine_v1.py` – Rule evaluation and risk summaries.
  - `migrate_heavy_to_E.py` – Moves heavy outputs from C: repo tree to E: roots.
  - `check_paths_and_write_readme.py` – Prints resolved roots and writes `docs/PATHS_CURRENT.md`.

---

## Centralized Paths and E: Drive Layout

Paths are centralized in `tools/path_config.py`. Heavy roots are configured in `configs/dataset.yaml` and prefer E:.

In `configs/dataset.yaml`:
- `dataset_root`: E:\safestride\datasets\ProcessedData
- `data_root`: E:\safestride\data\raw
- `work_root`: E:\safestride\data\working
- `out_root`: E:\safestride\out
- `release_root`: E:\safestride\release
- `log_root`: E:\safestride\logs
- `doc_root`: E:\safestride\docs

The `tools/path_config.py` module exposes these as `DATA_ROOT`, `WORK_ROOT`, `OUT_ROOT`, `RELEASE_ROOT`, `LOG_ROOT`, `DOC_ROOT`, and ensures directories exist.

Quick sanity check:
- PowerShell:
  - ` .\.venv\Scripts\python.exe tools\check_paths_and_write_readme.py`
  - Writes: `E:\safestride\docs\PATHS_CURRENT.md`

---

## Environment Setup (Windows)

- Recommended: Python 3.10–3.11 in a virtual environment under `.venv` at repo root.
- Typical setup:
  - `py -3.11 -m venv .venv`
  - `.\.venv\Scripts\pip install -r requirements.txt`
- Runners/scripts assume `.venv` at repo root; if absent, they fall back to `sys.executable`.

---

## Core Workflows

### 1) Inventory IMU Headers
- Purpose: Inspect real headers to inform normalization mapping.
- Command:
  - ` .\.venv\Scripts\python.exe tools\inventory_imu_headers.py`
- Output:
  - `docs/validation/imu_schema_raw.csv`

### 2) Normalize IMU Schema
- Auto-invoked by runners; standalone usage:
  - `tools/normalize_imu_schema.py` provides `normalize_file(path)` for canonical IMU DataFrame.
- Logs:
  - `E:\safestride\logs\imu_schema_map.log`
- CSV mapping table:
  - `E:\safestride\docs\validation\imu_schema_map.csv`

### 3) Internal Validation (MVP + Risk, resume-safe)
- Uses `configs/dataset.yaml: dataset_root` (E:) unless `--in_root` is specified.
- Optional cap to keep runs tight on lower-spec machines:
  - PowerShell env var: ` $env:SAFESTRIDE_CAP_N=200`
- Command:
  - ` .\.venv\Scripts\python.exe scripts\run_internal_validation.py --resume`
- Outputs:
  - `docs/validation/mvp_internal_summary.csv`
  - `docs/validation/risk_engine_internal_summary.csv`
  - `docs/validation/VALIDATION_REPORT_MVP_RISK_v1.md`
- Logs:
  - `E:\safestride\logs\validation_mvp_risk.log` (exclusion reasons, pipeline progress)

### 4) MVP on New Data (optionally run kinematics + risk)
- Command:
  - ` .\.venv\Scripts\python.exe scripts\run_safestride_mvp.py --in_root <folder> --resume --run_kin --run_risk`
- Auto-discovers frozen model from `release/models` or `E:\safestride\out_grid`.
- Writes per-trial outputs under `OUT_ROOT` (E:\safestride\out) and consolidated docs under `DOC_ROOT`.

### 5) Kinematic Surrogates Only
- Command:
  - ` .\.venv\Scripts\python.exe tools\build_kinematic_surrogates.py --in_root <folder> --pred_root <out-root>`
- Default outputs:
  - `E:\safestride\docs\vNext_multisignal\kin_surrogates.csv`

### 6) Migrate Heavy Artifacts to E:
- Purpose: Free C: space; enforce E: layout.
- Command:
  - ` .\.venv\Scripts\python.exe tools\migrate_heavy_to_E.py`
- Log and summary:
  - `E:\safestride\logs\migrate_heavy_to_E.log`

---

## Evidence Enforcement and Frozen Models

- Do not retrain during validation. Runners use frozen baselines.
- Evidence gate:
  - `tools/evidence_registry.py --enforce` is invoked by runners. If documentation is incomplete, it logs a FAIL but continues the pipeline while recording provenance.
- Thresholds include `source` and `evidence_grade` keys and are validated into docs (provenance ledgers).

---

## Determinism and Resume-Safe Behavior

- All pipelines skip completed work when `--resume` is used, and avoid non-deterministic operations.
- Normalization is deterministic; ambiguous or missing columns fail fast with explicit reasons.
- Time handling prefers `time_s`; if missing and `fs_hz` is known (from `configs/dataset.yaml`), it will synthesize `time_s` deterministically.

---

## Logs and Reports

- Logs (E:): `E:\safestride\logs`
  - `validation_mvp_risk.log` – inclusions/exclusions and pipeline errors.
  - `imu_schema_map.log` – normalization mapping trace.
  - `risk_engine.log`, `kinematics_engine.log`, `migrate_heavy_to_E.log`.
- Reports and CSVs (E:): `E:\safestride\docs`
  - Validation summaries, risk summaries, kinematic surrogates, and the validation report markdown.

---

## Troubleshooting

- Zero eligible trials:
  - Check `docs/validation/imu_schema_raw.csv` and `imu_schema_map.csv` for header alignment.
  - Ensure `time_s` exists or `fs_hz` is configured in `configs/dataset.yaml`.
  - See exclusion counts in `validation_mvp_risk.log` and `VALIDATION_REPORT_MVP_RISK_v1.md`.
- Disk space or slow runs:
  - Run `tools/migrate_heavy_to_E.py`.
  - Set `SAFESTRIDE_CAP_N` to limit eligible trials temporarily and use `--resume` to continue later.
- Model not found:
  - Ensure `release/models` or `E:\safestride\out_grid` contains the frozen model (e.g., HGB@300).

---

## Frequently Used Commands (copy/paste)

- Print paths and write a path README:
  - ` .\.venv\Scripts\python.exe tools\check_paths_and_write_readme.py`
- Inventory IMU headers:
  - ` .\.venv\Scripts\python.exe tools\inventory_imu_headers.py`
- Internal validation with cap:
  - ` $env:SAFESTRIDE_CAP_N=200; .\.venv\Scripts\python.exe scripts\run_internal_validation.py --resume`
- MVP over a folder with kinematics + risk:
  - ` .\.venv\Scripts\python.exe scripts\run_safestride_mvp.py --in_root E:\safestride\datasets\ProcessedData --resume --run_kin --run_risk`
- Migrate to E: roots:
  - ` .\.venv\Scripts\python.exe tools\migrate_heavy_to_E.py`

---

## Glossary

- IMU: Inertial Measurement Unit (accelerometer + gyroscope).
- vGRF: Vertical Ground Reaction Force.
- Resume-safe: Can stop/restart without losing progress.
- Evidence enforcement: Validation that thresholds/models document their provenance.

---

## Contributing Notes

- Keep source code on C:, heavy artifacts on E: per `path_config`.
- Prefer deterministic, logged behavior; avoid changing frozen models or thresholds in validation scripts.
- Add new tools under `tools/` and wire them to `path_config` roots.

---

If you get lost, start by checking `docs/PATHS_CURRENT.md` and reading this README top-to-bottom. Then try the internal validation with a small cap, inspect the outputs under `docs/validation/`, and iterate.
