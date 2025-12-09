# SafeStride vNext Roadmap

## Milestones

- **Per-task PI scaling (analysis → optional apply)**
  - Target: global 80% PI coverage ≥ 0.80 and per-task coverage ≥ 0.75 at 80% nominal.
  - Guardrail: No degradation to HGB@300 KPIs (median, p95, max unchanged within drift bands).
  - Outputs: `docs/uncertainty_autocal_per_task.json`, `docs/uncertainty_coverage_pre_post_per_task.csv`.
  - Apply: Off by default; optional per-task application controlled by orchestrator switch in a later run.

- **Dataset expansion (Gautier)**
  - Add to evidence registry with DOI: 10.1038/s41597-023-02077-3.
  - Record license and sampling rates in registry; confirm from the source.
  - Ingestion stub ready with schema mapping; no downloads by default.
  - Evidence enforcement remains PASS.
  - Note (current status): DOI recorded; license=TBD, fs_hz=TBD pending confirmation from the paper/portal.

- **Heteroscedastic / Conformal PIs (research track)**
  - Build residual maps and/or conformalized quantile regression prototypes.
  - Do not alter production predictions in vNext; diagnostics only.

- **Field validation plan (draft)**
  - Protocol: subject-wise GroupKFold holdouts; specify per-site data collection and QC.
  - Acceptance thresholds: replicate ≥ Clinical v1 KPIs, PI coverage ≥ 0.75 per task, ≥ 0.80 overall.

## Acceptance Gates

- **Per-task scaling acceptance**: global coverage ≥ 0.80; each task ≥ 0.75 (80% nominal).
- **Dataset integration**: registry updated with DOI, license, fs_hz; `tools/evidence_registry.py --enforce` remains PASS.
- **Research tracks**: prototyped and documented; no production changes.

## Notes

- Idempotent and resume-safe runs only; no model retraining.
- Clinical v1 artifacts remain frozen at `release/SafeStride_Clinical_v1_20251024.zip`.
