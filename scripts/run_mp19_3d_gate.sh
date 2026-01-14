#!/usr/bin/env bash
set -euo pipefail

echo "=== MP19 3D GATE START ==="

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
echo "REPO_ROOT=${REPO_ROOT}"
cd "${REPO_ROOT}"

pwd
ls -la | head -n 10 || true
git rev-parse --show-toplevel || true
git status -uno --porcelain=v1 || true
git show -s --oneline HEAD || true

echo "=== STEP 0: DEVICE PROBE ==="
python3 - << 'PY'
try:
    import torch
except Exception as e:  # pragma: no cover
    print("torch_import_error", repr(e))
else:
    has_mps = hasattr(torch.backends, "mps")
    avail = bool(has_mps and torch.backends.mps.is_available())
    built = bool(has_mps and torch.backends.mps.is_built())
    print("mps_available:", avail)
    print("mps_built:", built)
PY

DEVICE="$(python3 << 'PY'
try:
    import torch
except Exception:
    print("cpu")
    raise SystemExit
has_mps = hasattr(torch.backends, "mps")
if has_mps and torch.backends.mps.is_built() and torch.backends.mps.is_available():
    print("mps")
else:
    print("cpu")
PY
)"
echo "DEVICE=${DEVICE}"

echo "=== STEP 0b: ENV / PYTHON RESOLUTION ==="
# Prefer explicit SAFESTRIDE_VENV_PY when available; fall back to the venv from .env.local.example,
# then to plain python3.
SAFESTRIDE_VENV_PY="${SAFESTRIDE_VENV_PY:-"$HOME/venvs/safestrideclean/bin/python"}"
if [ ! -x "${SAFESTRIDE_VENV_PY}" ]; then
  echo "WARNING: SAFESTRIDE_VENV_PY executable not found at ${SAFESTRIDE_VENV_PY}, falling back to python3"
  SAFESTRIDE_VENV_PY="python3"
fi
export SAFESTRIDE_VENV_PY
echo "SAFESTRIDE_VENV_PY=${SAFESTRIDE_VENV_PY}"

mkdir -p data/vnext_gt_real_out

echo "=== STEP 1: LOCATE BASE CONFIG ==="
CONFIG_BASE="$(find configs -maxdepth 1 -type f -name 'vnext_example.yaml' 2>/dev/null | head -n 1 || true)"
if [ -z "${CONFIG_BASE}" ]; then
  CONFIG_BASE="$(find . -maxdepth 3 -type f \( -name '*.yaml' -o -name '*.yml' \) 2>/dev/null | grep -i 'vnext' | head -n 1 || true)"
fi
if [ -z "${CONFIG_BASE}" ]; then
  CONFIG_BASE="$(find . -maxdepth 5 -type f \( -name '*.yaml' -o -name '*.yml' \) 2>/dev/null | head -n 1 || true)"
fi

if [ -z "${CONFIG_BASE}" ]; then
  echo "MP19_ERROR: Could not locate a base YAML config."
  exit 1
fi

echo "CONFIG_BASE=${CONFIG_BASE}"
if command -v head >/dev/null 2>&1; then
  head -n 60 "${CONFIG_BASE}" || true
else
  sed -n '1,60p' "${CONFIG_BASE}" || true
fi

echo "=== STEP 2: BUILD 3D CANDIDATE CONFIG ==="
TMP_CFG="data/vnext_gt_real_out/tmp_candidate3d_20.yaml"

"${SAFESTRIDE_VENV_PY}" - "${CONFIG_BASE}" "${TMP_CFG}" << 'PY'
import sys, copy
from pathlib import Path

try:
    import yaml
except Exception as e:  # pragma: no cover
    print("YAML_IMPORT_ERROR", repr(e))
    raise

base = Path(sys.argv[1])
out = Path(sys.argv[2])

with base.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}

cfg = copy.deepcopy(cfg)

training = cfg.setdefault("training", {}) or {}
training["epochs"] = 20
training["lr"] = 1e-3
training["batch_size"] = 8
training["window_size"] = 256
training["window_stride"] = 128
training["target_norm"] = "none"

model = cfg.setdefault("model", {}) or {}
model["type"] = "grf3d"
model["grf_axes"] = "3d"
# target_grf_column is Fz-only; ensure it does not conflict with 3D
model.pop("target_grf_column", None)

paths = cfg.setdefault("paths", {}) or {}
paths.setdefault("data_root", "data/vnext_gt_real")
paths.setdefault("out_root", "data/vnext_gt_real_out")

out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("FINAL_MODEL_TYPE", model.get("type"))
print("FINAL_MODEL_GRF_AXES", model.get("grf_axes"))
PY

echo "=== STEP 2b: DRY IMPORT CONFIG ==="
"${SAFESTRIDE_VENV_PY}" - << 'PY'
from pathlib import Path
from vnext.core.config import load_config
from vnext.core.validation import validate_config

cfg_path = Path("data/vnext_gt_real_out/tmp_candidate3d_20.yaml")
print("DRY_IMPORT_CONFIG_PATH", cfg_path.as_posix(), "EXISTS", cfg_path.exists())

cfg = validate_config(load_config(cfg_path))
print("DRY_IMPORT_MODEL_TYPE", cfg.get("model", {}).get("type"))
print("DRY_IMPORT_GRF_AXES", cfg.get("model", {}).get("grf_axes"))
PY

echo "=== STEP 3: TRAIN 3D CANDIDATE (20 EPOCHS) ==="
"${SAFESTRIDE_VENV_PY}" scripts/train_vnext.py \
  --config "${TMP_CFG}" \
  --device "${DEVICE}"

echo "=== STEP 3b: LOCATE NEWEST RUN DIR WITH ARTIFACTS ==="
CANDIDATE_3D_RUN_DIR="$(
python3 << 'PY'
from pathlib import Path

root = Path("data") / "vnext_gt_real_out"
cands = []
for p in root.rglob("config.yaml"):
    run = p.parent
    if (run / "model_best.pt").exists() and (run / "train_history.csv").exists():
        try:
            mtime = run.stat().st_mtime
        except OSError:
            continue
        cands.append((mtime, run))

cands.sort(reverse=True, key=lambda x: x[0])
print(cands[0][1].as_posix() if cands else "")
PY
)"

echo "CANDIDATE_3D_RUN_DIR=${CANDIDATE_3D_RUN_DIR}"
if [ -z "${CANDIDATE_3D_RUN_DIR}" ]; then
  echo "MP19_ERROR: No candidate 3D run dir found after training."
  exit 1
fi

echo "=== STEP 4: EVAL + ANALYZE 3D CANDIDATE ==="
"${SAFESTRIDE_VENV_PY}" scripts/eval_vnext.py \
  --config "${CANDIDATE_3D_RUN_DIR}/config.yaml" \
  --run-dir "${CANDIDATE_3D_RUN_DIR}" \
  --checkpoint best \
  --save-preds \
  --device "${DEVICE}" \
  --analyze-after-eval \
  --analysis-out-dir "${CANDIDATE_3D_RUN_DIR}/analysis"

echo "=== STEP 4b: 3D GATE ARTIFACTS + METRICS ==="
GATE_SUMMARY="$(
CANDIDATE_3D_RUN_DIR="${CANDIDATE_3D_RUN_DIR}" python3 << 'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ["CANDIDATE_3D_RUN_DIR"])

problems = []

if not (run_dir / "model_best.pt").exists():
    problems.append("model_best.pt missing")

metrics_path = run_dir / "eval" / "eval_metrics_val.json"
if not metrics_path.exists():
    problems.append("eval_metrics_val.json missing")

preds_dir = run_dir / "eval" / "preds"
npz_files = list(preds_dir.glob("*.npz")) if preds_dir.exists() else []
if not npz_files:
    problems.append("no npz preds in eval/preds")

analysis_dir = run_dir / "analysis"
metrics_jsons = list(analysis_dir.glob("*metrics_summary*.json")) if analysis_dir.exists() else []
if not metrics_jsons:
    problems.append("no analysis metrics_summary json present")

val_rmse_mean = None
per_axis = {}
if metrics_path.exists():
    with metrics_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    m = payload.get("metrics") or {}
    v = m.get("rmse_mean")
    if v is not None:
        val_rmse_mean = float(v)
    per_axis = m.get("rmse_per_axis") or {}

if metrics_jsons:
    # Inspect Fz temporal lag if present in 3D analyzer summary
    with metrics_jsons[0].open("r", encoding="utf-8") as f:
        a = json.load(f)
    axis_summaries = a.get("axis_summaries") or {}
    fz_info = axis_summaries.get("Fz") or axis_summaries.get("fz")
    if isinstance(fz_info, dict):
        tl = fz_info.get("temporal_lag")
        if isinstance(tl, dict) and tl.get("status") == "FAIL":
            problems.append("Fz temporal lag FAILED in 3D analyzer")

if val_rmse_mean is None:
    problems.append("val_rmse_mean missing from eval_metrics_val.json")

gate_ok = not problems
reason = "ok" if gate_ok else "; ".join(problems)

out = {
    "run_dir": run_dir.as_posix(),
    "val_rmse_mean": float(val_rmse_mean) if val_rmse_mean is not None else None,
    "per_axis": per_axis,
    "gate_ok": bool(gate_ok),
    "reason": reason,
}
print(json.dumps(out))
PY
)"

echo "GATE_SUMMARY=${GATE_SUMMARY}"
export GATE_SUMMARY

GATE_OK="$(python3 << 'PY'
import json, os
s = os.environ.get("GATE_SUMMARY", "")
try:
    j = json.loads(s)
except Exception:
    print("NO")
else:
    print("YES" if j.get("gate_ok") else "NO")
PY
)"

GATE_REASON="$(python3 << 'PY'
import json, os
s = os.environ.get("GATE_SUMMARY", "")
try:
    j = json.loads(s)
except Exception:
    print("unknown")
else:
    print(str(j.get("reason", "unknown")))
PY
)"

VAL_RMSE="$(python3 << 'PY'
import json, os
s = os.environ.get("GATE_SUMMARY", "")
try:
    j = json.loads(s)
except Exception:
    print("nan")
else:
    v = j.get("val_rmse_mean")
    print("nan" if v is None else str(v))
PY
)"

PER_AXIS_STR="$(python3 << 'PY'
import json, os
s = os.environ.get("GATE_SUMMARY", "")
try:
    j = json.loads(s)
except Exception:
    print("{}")
else:
    pa = j.get("per_axis") or {}
    print(json.dumps(pa, sort_keys=True))
PY
)"

echo "VAL_RMSE_MEAN=${VAL_RMSE}"
echo "VAL_RMSE_PER_AXIS=${PER_AXIS_STR}"
echo "GATE_OK=${GATE_OK}"
echo "GATE_REASON=${GATE_REASON}"

echo "=== STEP 5: COMMIT 3D GATE CODE CHANGES (IF ANY) ==="
echo "GIT_STATUS_BEFORE_COMMIT:"
git status -uno --porcelain=v1 || true

# Only stage the 3D gate-related code files; avoid data and other unrelated changes.
git add scripts/analyze_3d_outputs.py scripts/eval_vnext.py scripts/run_mp19_3d_gate.sh 2>/dev/null || true

if git diff --cached --quiet; then
  echo "No code changes staged; skipping commit."
else
  git commit -m "add 3d eval+analysis gate" || true
fi

echo "GIT_STATUS_AFTER_COMMIT:"
git status -uno --porcelain=v1 || true
git diff --stat || true
git show -s --oneline HEAD || true

# FINAL REQUIRED OUTPUT LINES (do not print anything after these)
LOG_PATH="data/vnext_gt_real_out/mp19_3d_gate.log"
echo "LOG_SAVED: ${LOG_PATH}"
echo "CANDIDATE_3D_RUN_DIR: ${CANDIDATE_3D_RUN_DIR}"
echo "3D_METRICS: val_rmse_mean=${VAL_RMSE}, per_axis=${PER_AXIS_STR}"
if [ "${GATE_OK}" = "YES" ]; then
  echo "3D_GATE_OK: YES"
  echo "NEXT_MOVE: vast_gpu_3d_sweep"
else
  echo "3D_GATE_OK: NO (${GATE_REASON})"
  echo "NEXT_MOVE: 3d_neighborhood_sweep_local"
fi
