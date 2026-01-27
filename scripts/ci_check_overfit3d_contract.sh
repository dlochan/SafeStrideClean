#!/usr/bin/env bash
set -euo pipefail

## DEFAULT_ANALYZER_JSON
# Allow caller to override; otherwise use known-good default path
ANALYZER_JSON="${ANALYZER_JSON:-data/vnext_gt_real_out/vnext_fz/20260119-033819_df2881d2/analysis_eval/3d_metrics_summary.json}"
export ANALYZER_JSON


# CI-friendly wrapper around the canonical 3D overfit contract.
# - Uses existing analyzer artifacts only (no training, no eval).
# - Fails fast if the canonical summary JSON is missing.
# - Delegates contract logic to scripts/check_overfit3d_contract.py.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"
cd "${ROOT}"

JSON_PATH="data/vnext_gt_real_out/vnext_fz/20260119-033819_df2881d2/analysis_eval/3d_metrics_summary.json"

if [ ! -f "${JSON_PATH}" ]; then
  echo "CI FAIL overfit3d_contract: missing analyzer JSON at ${JSON_PATH}" >&2
  exit 1
fi

echo "[ci_check_overfit3d_contract] Using analyzer JSON: ${JSON_PATH}" >&2

python3 scripts/check_overfit3d_contract.py

echo "CI PASS overfit3d_contract"

# --- non-regression vs baseline ---
BASELINE_JSON="tests/baselines/overfit3d_contract_baseline.json"
echo "--- non-regression vs baseline ---"
python3 scripts/check_overfit3d_nonregression.py "$ANALYZER_JSON" "$BASELINE_JSON"
echo "PASS non-regression"
