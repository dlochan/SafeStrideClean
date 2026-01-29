#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Run PSU bundler and capture its output so we can extract OUT_DIR.
TMP_LOG="$(mktemp)"
trap 'rm -f "$TMP_LOG"' EXIT

bash scripts/psu_bundle_and_verify.sh | tee "$TMP_LOG"

OUT_DIR="$(grep '^OUT_DIR=' "$TMP_LOG" | tail -n 1 | sed 's/^OUT_DIR=//')"

if [ -z "$OUT_DIR" ] || [ ! -d "$OUT_DIR" ]; then
  echo "CI FAIL psu_bundle_contract: could not determine OUT_DIR" >&2
  exit 1
fi

python3 scripts/check_psu_bundle_nonregression.py \
  --baseline tests/baselines/psu_bundle_contract_baseline.json \
  --out-dir "$OUT_DIR"

echo "CI PASS psu_bundle_contract"
