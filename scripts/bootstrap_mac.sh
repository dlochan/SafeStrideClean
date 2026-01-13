#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"
VENV_DIR="$HOME/venvs/safestrideclean"
VENV_PY="$VENV_DIR/bin/python"

echo "[bootstrap_mac] repo root: $REPO_ROOT"

# --- Select Python >= 3.10 ---
PY_CANDIDATES=(python3.12 python3.11 python3.10 python3)
PY_SEL=""
for cand in "${PY_CANDIDATES[@]}"; do
  if command -v "$cand" >/dev/null 2>&1; then
    PY_SEL="$cand"
    break
  fi
done

if [[ -z "$PY_SEL" ]]; then
  echo "[bootstrap_mac] ERROR: Could not find a suitable Python interpreter (tried python3.12, python3.11, python3.10, python3)."
  echo "[bootstrap_mac] Install a newer Python via Homebrew and re-run:"
  echo "  brew install python@3.12"
  echo "  python3.12 --version"
  exit 2
fi

PY_VER_SEL="$($PY_SEL -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
if ! $PY_SEL - << 'PY'
import sys
sys.exit(0 if (sys.version_info.major, sys.version_info.minor) >= (3, 10) else 1)
PY
then
  echo "[bootstrap_mac] ERROR: Need Python >= 3.10. Detected $PY_SEL = $PY_VER_SEL."
  echo "[bootstrap_mac] Install a newer Python via Homebrew and re-run:"
  echo "  brew install python@3.12"
  echo "  python3.12 --version"
  exit 2
fi

echo "[bootstrap_mac] using interpreter: $PY_SEL ($PY_VER_SEL)"

mkdir -p "$(dirname "$VENV_DIR")"
if [[ ! -d "$VENV_DIR" ]]; then
  echo "[bootstrap_mac] creating venv at $VENV_DIR"
  "$PY_SEL" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

PY_VER="$($VENV_PY -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"
echo "[bootstrap_mac] python: $PY_VER (venv: $VENV_DIR)"

cd "$REPO_ROOT"

if [[ -f "pyproject.toml" || -f "setup.py" ]]; then
  echo "[bootstrap_mac] installing editable package (pip install -e .)"
  "$VENV_PY" -m pip install -U pip
  "$VENV_PY" -m pip install -e .
elif ls requirements*.txt >/dev/null 2>&1; then
  REQ_FILE="$(ls requirements*.txt | head -n1)"
  echo "[bootstrap_mac] installing requirements from $REQ_FILE"
  "$VENV_PY" -m pip install -U pip
  "$VENV_PY" -m pip install -r "$REQ_FILE"
else
  echo "[bootstrap_mac] WARNING: no pyproject.toml/setup.py or requirements*.txt found; skipping dependency install"
fi

"$VENV_PY" -m pip install -U numpy pandas pyyaml torch >/dev/null 2>&1 || true

"$VENV_PY" - << 'PY'
import importlib

required = ["numpy", "pandas", "yaml", "torch"]
failed = []
for m in required:
    try:
        importlib.import_module(m)
    except Exception as e:
        failed.append(f"{m}: {type(e).__name__}: {e}")

if failed:
    raise SystemExit("Import check failed:\n" + "\n".join(failed))

try:
    importlib.import_module("matplotlib")
except Exception:
    print("[bootstrap_mac] WARNING: matplotlib not installed; continuing without it.")
else:
    print("[bootstrap_mac] OK: matplotlib import passed")
PY

if [[ -z "${SAFESTRIDE_DATA_ROOT:-}" ]]; then
  echo "[bootstrap_mac] ERROR: SAFESTRIDE_DATA_ROOT is not set."
  echo "Set it to the directory that contains 'datasets/ProcessedData', for example:"
  echo "  export SAFESTRIDE_DATA_ROOT=\"/Volumes/Extreme SSD/safestride_data\""
  exit 2
fi

if [[ ! -d "$SAFESTRIDE_DATA_ROOT" ]]; then
  echo "[bootstrap_mac] ERROR: SAFESTRIDE_DATA_ROOT does not exist: $SAFESTRIDE_DATA_ROOT"
  exit 2
fi

echo "[bootstrap_mac] SAFESTRIDE_DATA_ROOT: $SAFESTRIDE_DATA_ROOT"

"$VENV_PY" -m py_compile \
  src/vnext/data/datasets.py \
  scripts/train_vnext.py \
  scripts/eval_vnext.py

SMOKE_CMD=("$VENV_PY" "scripts/import_smoke_test.py")
echo "[bootstrap_mac] running smoke: ${SMOKE_CMD[*]}"
if ! "${SMOKE_CMD[@]}"; then
  echo "[bootstrap_mac] ERROR: smoke command failed: ${SMOKE_CMD[*]}"
  exit 3
fi

echo "[bootstrap_mac] OK: bootstrap complete"
