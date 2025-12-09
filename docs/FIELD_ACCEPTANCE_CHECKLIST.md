# Field Acceptance Checklist

Use this checklist to validate a field batch before sharing results.

- Evidence enforcement
  - Run: `.\.venv\Scripts\python.exe .\tools\evidence_registry.py --enforce`
  - PASS required. If FAIL, fix docs/evidence/dataset_registry.csv and configs/clinical_thresholds.yaml sources/grades.

- Input sanity
  - Input root exists and contains `<trial>_imu_real.csv` files.
  - Time column `time_s` numeric and monotonic; sampling ~20–2000 Hz.

- Processing gates
  - Success rate ≥ 0.90 (processed/eligible trials).
  - Quality OK rate ≥ 0.90 (from docs/clinical_scores_summary.csv for processed trials).
  - Baseline KPI (HGB@300) for processed trials: median ≤ 10 %BW, p95 ≤ 20 %BW.
  - If truth files exist in data/working, PI coverage at 0.80 nominal ≥ 0.75. Otherwise “coverage unknown” (acceptable).

- Outputs present
  - E:\safestride\out_field\leaderboard_field.csv
  - docs/clinical_packs_field/*.pdf (one per subject)
  - docs/field_acceptance.txt
  - release/SafeStride_FieldValidation_<YYYYMMDD>.zip

- Troubleshooting
  - Zero windows: pipeline falls back 300→200→100 ms.
  - Missing packs: ensure clinical_scores_summary.csv has rows for those trials; rerun.
  - Evidence FAIL: fill doi_or_url, license, fs_hz for the dataset entry; ensure thresholds sources/evidence_grade set.
