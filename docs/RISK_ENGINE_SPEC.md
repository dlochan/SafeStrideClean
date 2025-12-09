# Risk Engine v1 (mechanical risk & workload, rule-based)

Purpose
- Provide interpretable, literature-linked risk indicators from Clinical_v1 vGRF outputs plus kinematic surrogates and longitudinal workload.
- No black-box injury prediction. All flags are exposure/risk indicators with uncertainty.

Inputs
- mvp/metrics_mvp.csv (peak vGRF %BW, loading rate, impulse, stance time)
- docs/vNext_multisignal/kin_surrogates.csv (knee/trunk/pelvis surrogates)
- Optional longitudinal rollups (already implicit via per-day aggregation in MVP)
- Thresholds: configs/clinical_thresholds.yaml::risk_engine_v1
- Evidence registry: docs/evidence/study_registry.csv

Core rules (initial set)
- high_load_low_flex:
  - peak vGRF %BW >= threshold AND knee_flex_ic_deg <= threshold.
  - Evidence: PMC5763249_LandingTasks, PubMed20303276_KneeFlexGRF.
- chronic_load_spike:
  - acute:chronic ratio from MVP > threshold.
  - Evidence: OARSI2020_WalkingImpact (loading-related symptoms/exposure).

Outputs
- docs/risk_engine/risk_features.csv (per-trial feature table: merges MVP metrics + kin surrogates)
- docs/risk_engine/risk_flags.csv (flag rows with metric, threshold, direction, evidence key, and uncertainty note)
- docs/risk_engine/RISK_ENGINE_REPORT.md (counts, brief interpretation)

Uncertainty
- When PI or RMSE context is unavailable, include uncertainty_note='no_pi_context'.
- Rules should be conservative and only raise flags when clearly above thresholds.

Constraints
- Resume-safe and idempotent. Logs to logs/risk_engine.log.
- Clinical_v1 remains frozen. This engine only reads its outputs and adds rule-based indicators.
