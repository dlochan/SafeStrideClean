#!/usr/bin/env bash
set -euo pipefail

# Helper script to run a full-manifest generalization evaluation for a
# completed vNext run directory. It creates a temporary eval-only config
# under <RUN_DIR>/eval_full/ that:
#   - points data.val_manifest to the full vnext_val_real manifest
#   - disables any subset_indices_path/subset_num_windows restrictions
# and then invokes eval_vnext via run_vnext.sh using that config.
#
# Usage:
#   ./scripts/run_generalization_eval.sh <RUN_DIR> [--device mps]
#
# Outputs:
#   - <RUN_DIR>/generalization_eval_full.log
#   - <RUN_DIR>/eval_full/config_eval_full.yaml
#   - <RUN_DIR>/eval_full/eval_metrics_val.json (copied from <RUN_DIR>/eval/)
#   - <RUN_DIR>/analysis_eval_full/ (if --analyze-after-eval succeeds)

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <RUN_DIR> [--device DEVICE] [extra eval_vnext args...]" >&2
  exit 1
fi

RUN_DIR="$1"
shift || true

if [ ! -d "$RUN_DIR" ]; then
  echo "RUN_DIR does not exist or is not a directory: $RUN_DIR" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_SRC="$RUN_DIR/config.yaml"
if [ ! -f "$CONFIG_SRC" ]; then
  echo "Config copy not found under run_dir: $CONFIG_SRC" >&2
  exit 1
fi

EVAL_FULL_DIR="$RUN_DIR/eval_full"
mkdir -p "$EVAL_FULL_DIR"
CONFIG_EVAL="$EVAL_FULL_DIR/config_eval_full.yaml"

VENV_PY="$ROOT/.venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo ".venv python not found at $VENV_PY" >&2
  exit 1
fi

# Mirror run_vnext.sh environment so that imports (including PyYAML)
# work correctly even when using -S.
export PYTHONPATH="$ROOT/src:$ROOT/.venv/lib/python3.9/site-packages${PYTHONPATH:+:$PYTHONPATH}"

# Create a temporary eval-only config that disables subset usage and
# forces the full validation manifest.
"$VENV_PY" -S - "$CONFIG_SRC" "$CONFIG_EVAL" << 'EOF'
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception as e:  # pragma: no cover - defensive
    print(f"Failed to import yaml: {e}", file=sys.stderr)
    raise SystemExit(1)

if len(sys.argv) != 3:
    print("Usage: create_eval_config.py <SRC> <DST>", file=sys.stderr)
    raise SystemExit(1)

src = Path(sys.argv[1])
dst = Path(sys.argv[2])

if not src.is_file():
    print(f"Source config not found: {src}", file=sys.stderr)
    raise SystemExit(1)

with src.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

data = cfg.get("data") or {}

# Disable any subset restrictions.
data.pop("subset_indices_path", None)
data["subset_num_windows"] = 0

# Force full validation manifest for generalization.
# This is relative to paths.data_root inside the config and matches
# the canonical overfit config semantics.
data["val_manifest"] = "manifests/vnext_val_real.csv"

cfg["data"] = data

dst.parent.mkdir(parents=True, exist_ok=True)
with dst.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
EOF

LOG_PATH="$RUN_DIR/generalization_eval_full.log"
echo "[run_generalization_eval] Starting full-manifest eval for RUN_DIR=$RUN_DIR" | tee "$LOG_PATH"

EXTRA_ARGS=("$@")

ANALYSIS_DIR="$RUN_DIR/analysis_eval_full"

EVAL_CMD=("$ROOT/run_vnext.sh" eval \
  --config "$CONFIG_EVAL" \
  --run-dir "$RUN_DIR" \
  --save-preds \
  --preds-suffix "eval_full" \
  --analyze-after-eval \
  --analysis-out-dir "$ANALYSIS_DIR" \
  "${EXTRA_ARGS[@]}")

printf '[run_generalization_eval] Command: ' | tee -a "$LOG_PATH"
printf '%q ' "${EVAL_CMD[@]}" | tee -a "$LOG_PATH"
echo | tee -a "$LOG_PATH"

# Run eval_vnext via run_vnext.sh and tee output into the log.
if ! "${EVAL_CMD[@]}" 2>&1 | tee -a "$LOG_PATH"; then
  echo "[run_generalization_eval] Eval command failed" | tee -a "$LOG_PATH" >&2
  exit 1
fi

# Copy the populated metrics JSON into eval_full/ for downstream tooling.
METRICS_SRC_VAL="$RUN_DIR/eval/eval_metrics_val.json"
METRICS_SRC_TEST="$RUN_DIR/eval/eval_metrics_test.json"
METRICS_SRC=""

if [ -f "$METRICS_SRC_VAL" ]; then
  METRICS_SRC="$METRICS_SRC_VAL"
elif [ -f "$METRICS_SRC_TEST" ]; then
  METRICS_SRC="$METRICS_SRC_TEST"
else
  echo "[run_generalization_eval] No eval_metrics_{val,test}.json found under $RUN_DIR/eval" | tee -a "$LOG_PATH" >&2
  exit 1
fi

METRICS_DST="$EVAL_FULL_DIR/$(basename "$METRICS_SRC")"
cp "$METRICS_SRC" "$METRICS_DST"

echo "[run_generalization_eval] Metrics JSON copied to $METRICS_DST" | tee -a "$LOG_PATH"

echo "[run_generalization_eval] DONE" | tee -a "$LOG_PATH"
