# FZ Baseline Report

Run dir: `data\vnext_gt_real_out\vnext_fz\20251212-014808_a72332cc`
Units: `unknown`
Windows: N=180, T=256

## Core metrics (window-level aggregates)

- RMSE: 325.6652 ± 109.2615
- MAE: 271.4767 ± 84.8971
- Bias: -82.5615 ± 63.4679
- Std(error): 310.6509 ± 103.1861
- nRMSE (RMSE/median(|y_true|)): 0.5638 ± 0.2158
- Pearson r: 0.7851 ± 0.0678 (finite windows=180)

## Failure mode checks

- magnitude_sanity: **PASS**
- constant_predictions: **PASS**
- temporal_lag: **PASS**
- over_smoothing: **PASS**
- mode_collapse: **PASS**

## Per-subject availability

- subject_id: unavailable (not present in manifest)
