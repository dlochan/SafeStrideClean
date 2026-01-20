#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$ROOT/.venv/lib/python3.9/site-packages"

if [ ! -d "$ROOT/.venv" ]; then
  echo ".venv not found under $ROOT" >&2
  exit 1
fi

mkdir -p "$TARGET"

PY="/Library/Developer/CommandLineTools/usr/bin/python3"

BASE_FLAGS=(
  -m pip install
  --target "$TARGET"
  --upgrade
  --no-warn-script-location
  --no-cache-dir
  --ignore-installed
  --no-deps
)

set +e
"$PY" "${BASE_FLAGS[@]}" numpy PyYAML || true
"$PY" "${BASE_FLAGS[@]}" pandas || true

if ! "$PY" "${BASE_FLAGS[@]}" torch; then
  "$PY" -m pip install \
    --target "$TARGET" \
    --upgrade \
    --no-warn-script-location \
    --no-cache-dir \
    --ignore-installed \
    --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    torch || true
fi
set -e

if "$ROOT/run_vnext.sh" import_check; then
  exit 0
else
  exit 1
fi
