#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$ROOT/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo ".venv python not found at $VENV_PY" >&2
  exit 1
fi
export PYTHONPATH="$ROOT/src:$ROOT/.venv/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"
echo "python_executable: $VENV_PY"
"$VENV_PY" -S -V
"$VENV_PY" -S - << "EOF"
import sys
import vnext, numpy, torch, pandas, yaml
print("vnext_path", vnext.__file__)
print("numpy_version", numpy.__version__)
print("torch_version", torch.__version__)
print("pandas_version", pandas.__version__)
print("yaml_version", yaml.__version__)
EOF
echo "train_vnext_help:"
"$VENV_PY" -S "$ROOT/scripts/train_vnext.py" --help | head -n 20
echo "eval_vnext_help:"
"$VENV_PY" -S "$ROOT/scripts/eval_vnext.py" --help | head -n 20
