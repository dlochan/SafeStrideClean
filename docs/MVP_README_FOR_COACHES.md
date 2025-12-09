# SafeStride MVP: What this does (and does not do)

What you get
- Vertical ground reaction force (vGRF) waveform from IMUs, with uncertainty bands.
- Simple, interpretable load metrics per session: peak load, loading rate, impulse, stance time.
- A small set of risk and workload indicators tied to literature (not diagnoses):
  - High impact loading
  - Stiff landings
  - Elevated left-right asymmetry (when both sides exist)
  - Acute:chronic workload spikes
  - High load with low knee flexion at contact (when kinematic surrogates are available)
- A short acceptance summary (ACCEPTANCE_MVP.txt) and a risk report.

What this is not
- Not an injury predictor. These are mechanical exposure indicators with uncertainty.
- Not a replacement for clinical judgement. Use in context with athlete history and goals.

How to run (PowerShell)
- powershell -NoProfile -ExecutionPolicy Bypass -File .\tools\run_mvp.ps1 -InRoot "E:\my_new_data" -Resume
- Outputs appear in:
  - mvp/leaderboard_mvp.csv, mvp/metrics_mvp.csv, mvp/flags_mvp.csv
  - docs/risk_engine/risk_features.csv, risk_flags.csv, RISK_ENGINE_REPORT.md
  - mvp/ACCEPTANCE_MVP.txt (quick summary for staff)

How to interpret
- Trends matter. Use acute:chronic ratio and asymmetry over time.
- Focus on change from baseline for each athlete.
- When uncertainty is high or data is sparse, be conservative and re-check with additional sessions.
