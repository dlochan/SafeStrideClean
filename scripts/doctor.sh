#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

echo "REPO_ID=safestride_clean"
echo "=== DOCTOR (repo) ==="
echo "PWD=$(pwd)"
echo "GIT_ROOT=$ROOT"
echo "BRANCH=$(git branch --show-current)"
echo "--- LAST 5 COMMITS ---"
git log -5 --oneline
echo "--- GIT STATUS (PORCELAIN) ---"
git status --porcelain=v1 | sed -n '1,200p' || true

echo "--- PYTHON ---"
python3 - <<'PY'
import sys
print("python3_ok=True", "version", sys.version.split()[0])
PY

echo "--- OPENPYXL ---"
python3 - <<'PY'
try:
    import openpyxl
    print("openpyxl_ok=True", "version", getattr(openpyxl, "__version__", "unknown"))
except Exception as e:
    print("openpyxl_ok=False", str(e))
PY

echo "========================="
