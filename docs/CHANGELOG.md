# CHANGELOG

## 2025-10-24 — Clinical v1 Acceptance + vNext Scope
- Clinical v1 accepted. KPIs (HGB@300): rows=345, median=7.019, p95=16.844, max=19.830.
- Uncertainty (80% PI): coverage=0.7908 (post-cal; gate ≥0.70).
- Evidence enforcement: PASS; TBD=0; provenance recorded in `docs/evidence/`.
- vNext scope queued:
  - Per-task PI scaling (analysis → optional apply) targeting global ≥0.80 and per-task ≥0.75 at 80% nominal.
  - Dataset expansion (Gautier) registered; ingestion stub to be added; enforcement to remain PASS.
  - Heteroscedastic/conformal PI prototypes (research track), no production changes.
