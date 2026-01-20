#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PY="$ROOT/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo ".venv python not found at $VENV_PY" >&2
  exit 1
fi

export PYTHONPATH="$ROOT/src:$ROOT/.venv/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"

_print_banner() {
  local subcommand="$1"; shift || true
  local config=""
  local device=""
  local args=("$@")
  local i=0

  while [ $i -lt ${#args[@]} ]; do
    case "${args[$i]}" in
      --config)
        if [ $((i + 1)) -lt ${#args[@]} ]; then
          config="${args[$((i + 1))]}"
        fi
        i=$((i + 2))
        ;;
      --device)
        if [ $((i + 1)) -lt ${#args[@]} ]; then
          device="${args[$((i + 1))]}"
        fi
        i=$((i + 2))
        ;;
      *)
        i=$((i + 1))
        ;;
    esac
  done

  local ts
  ts="$(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "[run_vnext] subcommand=${subcommand} ts=${ts} config=${config:-unknown} device=${device:-unknown} args=${args[*]}"
}

COMMAND="${1:-}"
if [ -z "$COMMAND" ]; then
  echo "Usage: $0 import_check|train|eval ..." >&2
  exit 1
fi
shift

case "$COMMAND" in
  import_check)
    "$VENV_PY" -S - << "EOF"
import sys
import vnext, numpy, torch, pandas, yaml
print("python_executable", sys.executable)
print("vnext_path", vnext.__file__)
print("numpy_version", numpy.__version__)
print("torch_version", torch.__version__)
print("pandas_version", pandas.__version__)
print("yaml_version", yaml.__version__)
EOF
    ;;
  train)
    _print_banner "train" "$@"
    exec "$VENV_PY" -S "$ROOT/scripts/train_vnext.py" "$@" 2>&1
    ;;
  eval)
    _print_banner "eval" "$@"
    exec "$VENV_PY" -S "$ROOT/scripts/eval_vnext.py" "$@" 2>&1
    ;;
  *)
    echo "Unknown subcommand: $COMMAND" >&2
    exit 1
    ;;
esac
