#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 RUN_DIR [--device DEVICE] [--mode read_only|generate]" >&2
  exit 1
fi

RUN_DIR="$1"
shift || true

DEVICE="mps"
MODE="read_only"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --device)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --device requires an argument" >&2
        exit 1
      fi
      DEVICE="$2"
      shift 2
      ;;
    --mode)
      if [ "$#" -lt 2 ]; then
        echo "ERROR: --mode requires an argument" >&2
        exit 1
      fi
      MODE="$2"
      shift 2
      ;;
    *)
      echo "ERROR: Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [ "$MODE" != "read_only" ] && [ "$MODE" != "generate" ]; then
  echo "ERROR: --mode must be 'read_only' or 'generate' (got '$MODE')" >&2
  exit 1
fi

if [ ! -d "$RUN_DIR" ]; then
  echo "ERROR: RUN_DIR '$RUN_DIR' does not exist or is not a directory" >&2
  exit 1
fi

RUN_DIR_ABS="$(cd "$RUN_DIR" && pwd)"

BUNDLES_DIR="$RUN_DIR/bundles"
TS="$(date +%Y%m%d-%H%M%S)"
BUNDLE_DIR="$BUNDLES_DIR/3d_eval_bundle_${TS}"

mkdir -p "$BUNDLE_DIR"

ANALYZER_64="$RUN_DIR/analysis_eval/3d_metrics_summary.json"
ANALYZER_FULL="$RUN_DIR/analysis_eval_full/3d_metrics_summary.json"
EVAL_METRICS="$RUN_DIR/eval_full/eval_metrics_val.json"

if [ "$MODE" = "generate" ]; then
  echo "[make_3d_eval_bundle] Running full-manifest eval for RUN_DIR=${RUN_DIR} with device=${DEVICE}..." >&2
  bash scripts/run_generalization_eval.sh "$RUN_DIR" --device "$DEVICE"
fi

for path in "$ANALYZER_64" "$ANALYZER_FULL" "$EVAL_METRICS"; do
  if [ ! -f "$path" ]; then
    echo "ERROR: expected artifact not found (mode=${MODE}): $path" >&2
    exit 1
  fi
done

cp "$ANALYZER_64" "$BUNDLE_DIR/analyzer_64.json"
cp "$ANALYZER_FULL" "$BUNDLE_DIR/analyzer_full.json"
cp "$EVAL_METRICS" "$BUNDLE_DIR/eval_metrics_val.json"

RUN_DIR_REL="$RUN_DIR"

RUN_DIR_ENV="$RUN_DIR_REL"
RUN_DIR_ABS_ENV="$RUN_DIR_ABS"
BUNDLE_DIR_ENV="$BUNDLE_DIR"

ANALYZER_64_BUNDLE="$BUNDLE_DIR/analyzer_64.json"
ANALYZER_FULL_BUNDLE="$BUNDLE_DIR/analyzer_full.json"
EVAL_METRICS_BUNDLE="$BUNDLE_DIR/eval_metrics_val.json"

RUN_DIR="$RUN_DIR_ENV" RUN_DIR_ABS="$RUN_DIR_ABS_ENV" BUNDLE_DIR="$BUNDLE_DIR_ENV" \
ANALYZER_64_BUNDLE="$ANALYZER_64_BUNDLE" ANALYZER_FULL_BUNDLE="$ANALYZER_FULL_BUNDLE" \
EVAL_METRICS_BUNDLE="$EVAL_METRICS_BUNDLE" \
python3 - << 'PY'
import json, os, subprocess

run_dir = os.environ["RUN_DIR"]
run_dir_abs = os.environ["RUN_DIR_ABS"]
bundle_dir = os.environ["BUNDLE_DIR"]
analyzer_64_path = os.environ["ANALYZER_64_BUNDLE"]
analyzer_full_path = os.environ["ANALYZER_FULL_BUNDLE"]
eval_metrics_path = os.environ["EVAL_METRICS_BUNDLE"]


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


a64 = load_json(analyzer_64_path)
afull = load_json(analyzer_full_path)
# Currently unused, but parsed for completeness and future extension
_eval_metrics = load_json(eval_metrics_path)

config_path = None
candidates = [
    os.path.join(run_dir, "config.yaml"),
    os.path.join(run_dir, "config.yml"),
    os.path.join(run_dir, "eval_full", "config.yaml"),
    os.path.join(run_dir, "eval_full", "config.yml"),
]
for c in candidates:
    if os.path.exists(c):
        config_path = c
        break

_git_hash = None
try:
    _git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=os.getcwd(),
        text=True,
    ).strip()
except Exception:
    _git_hash = None


def axis(summary: dict, name: str):
    s = summary.get("axis_summaries", {}).get(name, {})
    return s.get("rmse"), s.get("corr")


units64 = a64.get("units_detected")
n64 = a64.get("num_windows")
len64 = a64.get("window_len")
fz_rmse_64, fz_corr_64 = axis(a64, "Fz")
gate_status = a64.get("gate", {}).get("status")

units_full = afull.get("units_detected")
n_full = afull.get("num_windows")
len_full = afull.get("window_len")
fx_rmse_f, fx_corr_f = axis(afull, "Fx")
fy_rmse_f, fy_corr_f = axis(afull, "Fy")
fz_rmse_f, fz_corr_f = axis(afull, "Fz")

lines = []
lines.append("# 3D Evaluation Bundle")
lines.append("")
lines.append("## Run metadata")
lines.append(f"- Run dir (relative): `{run_dir}`")
lines.append(f"- Run dir (absolute): `{run_dir_abs}`")
if config_path:
    lines.append(f"- Config: `{config_path}`")
if _git_hash:
    lines.append(f"- Git commit: `{_git_hash}`")
lines.append("")
lines.append("## 64-window overfit subset")
lines.append(f"- units_detected: `{units64}`")
lines.append(f"- num_windows: `{n64}`")
lines.append(f"- window_len: `{len64}`")
lines.append(f"- Fz_rmse: `{fz_rmse_64}`")
lines.append(f"- Fz_corr: `{fz_corr_64}`")
lines.append(f"- gate.status: `{gate_status}`")
lines.append("")
lines.append("## Full-manifest evaluation")
lines.append(f"- units_detected: `{units_full}`")
lines.append(f"- num_windows: `{n_full}`")
lines.append(f"- window_len: `{len_full}`")
lines.append(f"- Fx_rmse: `{fx_rmse_f}` (corr `{fx_corr_f}`)")
lines.append(f"- Fy_rmse: `{fy_rmse_f}` (corr `{fy_corr_f}`)")
lines.append(f"- Fz_rmse: `{fz_rmse_f}` (corr `{fz_corr_f}`)")
lines.append("")
lines.append("## Interpretation (descriptive only)")
lines.append("- The 64-window subset summarizes how the model performs on the fixed canonical validation slice.")
lines.append("- The full-manifest metrics describe average 3D GRF performance over all windows in the canonical validation manifest.")
lines.append("- Comparing subset vs full-manifest Fz_rmse/Fz_corr shows how tightly the model fits the subset versus generalizes to the full manifest.")
lines.append("- Use this bundle alongside experiment notes to judge whether this run is appropriate for downstream biomechanics or monitoring use.")

out_path = os.path.join(bundle_dir, "bundle_report.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
PY

echo "BUNDLE_DIR=${BUNDLE_DIR}"
