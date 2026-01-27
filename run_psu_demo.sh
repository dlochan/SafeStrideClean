#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

echo "=== PSU DEMO (minimal) ==="
echo "HEAD=$(git rev-parse --short HEAD)"
echo "BRANCH=$(git branch --show-current)"
echo "Running CI contract check + non-regression..."
bash scripts/ci_check_overfit3d_contract.sh
echo "PSU DEMO: PASS"
