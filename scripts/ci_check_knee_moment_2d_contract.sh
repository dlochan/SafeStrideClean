#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 scripts/check_knee_moment_2d_nonregression.py \
  --mode check \
  --baseline tests/baselines/knee_moment_2d_contract_baseline.json \
  --print_regen_cmd || exit $?

echo "CI PASS knee_moment_2d_contract"
