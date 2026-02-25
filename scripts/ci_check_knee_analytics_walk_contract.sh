#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

out_dir_line="$(python3 scripts/run_knee_analytics_walking.py | tail -n 1)"
if [[ "$out_dir_line" != OUT_DIR=* ]]; then
  echo "FAIL knee_analytics_walk_contract: runner did not print OUT_DIR" >&2
  echo "$out_dir_line" >&2
  exit 1
fi
OUT_DIR="${out_dir_line#OUT_DIR=}"

python3 scripts/check_knee_analytics_nonregression.py \
  --mode check \
  --out-dir "$OUT_DIR" \
  --baseline tests/baselines/knee_analytics_walk_contract_baseline.json \
  --print_regen_cmd || exit $?

echo "CI PASS knee_analytics_walk_contract"
