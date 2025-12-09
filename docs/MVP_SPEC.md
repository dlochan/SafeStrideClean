# SafeStride MVP (v0) Contract

This document defines a minimal, reproducible SafeStride MVP that a non-expert can run on new IMU data. Clinical_v1 remains FROZEN; this MVP only adds wrappers and reporting on top of the frozen vertical GRF model and the evidence engine.

## Inputs
- Sensors: dual IMUs (thigh–shank or knee-pair), CSV per trial.
- Required metadata per trial:
  - `bw_kg` (body mass in kilograms; fallback allowed but must be printed when assumed)
  - `task` tag (e.g., normal, running)
  - `limb` or side in trial name when applicable (e.g., left/right)
  - `session_id` (derived from folder/date if not provided)
- Sampling: IMU resampled/handled by existing pipeline; predictions assume 200 Hz unless timestamps are present.

## Core Outputs (v0)
- Vertical GRF waveform (%BW) with 80% prediction interval (PI) band.
- Primary metrics computed per trial:
  - Peak vGRF %BW
  - Loading rate (reported in %BW/s and N/s when BW is known)
  - Impulse (%BW·s; integrated positive vGRF during stance)
  - Stance time (s; duration with vGRF > 5% BW; longest continuous interval if multiple)
  - Left–right asymmetry index (AI = |L−R|/max(L,R)) when a paired side exists in the same subject/session/task
  - Longitudinal load index (simple acute:chronic ratio using daily summed impulse/peaks; 7d vs 28d windows)

## Clinical Flags (risk indicators, not diagnoses)
- High impact loading
- Stiff landing
- Elevated asymmetry
- Chronic load high

For each flag:
- Apply thresholds defined in `configs/clinical_thresholds.yaml` under the `mvp` section.
- Each threshold row includes: threshold value, direction (high/low/both), `source` key (mapped to docs/evidence/study_registry.csv), and `evidence_grade`.

Strict rule: flags are risk indicators only.

## Evidence & Uncertainty
- Use FROZEN Clinical_v1 models only (no retraining).
- Enforce evidence via `tools/evidence_registry.py --enforce` prior to MVP run.
- Uncertainty: reuse calibrated PI via existing `tools/add_uncertainty.py` and `docs/vNext_multisignal/pi_calibration.json` if present. When not available, document defaults.

## Runner Contract
- `scripts/run_safestride_mvp.py` (resume-safe):
  - Input root of IMU CSVs + optional per-trial metadata.
  - Standardizes inputs via existing adapters, predicts vGRF using frozen Clinical_v1 model, adds uncertainty, computes metrics, applies MVP thresholds, and writes:
    - `mvp/leaderboard_mvp.csv`
    - `mvp/metrics_mvp.csv`
    - `mvp/flags_mvp.csv`
  - Resume behavior: skip trials with existing outputs unless `--force`.

## vNext (non-MVP, optional enhancements)
- Multi-signal sandbox (3D GRF, kinematics, richer asymmetry) can plug in later using the same runner structure.
- Do not change existing MultiSignal_v1 or Grouvel sandbox bundles. Do not claim production readiness for 3D until RMSE and coverage meet acceptable ranges.

## Coach Workflow (turnkey)
- Drop IMU CSVs into a folder (per trial). Supported names: `*_imu_real.csv`, `*_imu.csv`, or `.csv.gz` variants.
- Run one command:
  - PowerShell: `powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_mvp.ps1 -InRoot "E:\my_new_data" -Resume`
- Read these files:
  - `mvp/leaderboard_mvp.csv` (overview)
  - `mvp/flags_mvp.csv` (per-trial flags)
  - `docs/risk_engine/RISK_ENGINE_REPORT.md` (mechanical risk profile)
  - `mvp/ACCEPTANCE_MVP.txt` (what ran, model used, quick stats)
