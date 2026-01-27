#!/usr/bin/env bash
set -euo pipefail

# Goal: make a clean demo bundle + verify the bundle contract, without relying on your current working tree.
# This assumes your repo already has the demo scripts (doctor.sh / run_psu_demo.sh / bundle contract checks).

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

ts="$(date +%Y%m%d-%H%M%S)"
OUT_DIR="artifacts/psu_demo_bundle_${ts}"
mkdir -p "$OUT_DIR"

echo "== PSU_BUNDLE start =="
echo "OUT_DIR=$OUT_DIR"
echo "HEAD=$(git rev-parse --short HEAD)"
echo "BRANCH=$(git branch --show-current)"

# 1) Repo doctor (fast)
if [ -f scripts/doctor.sh ]; then
  bash scripts/doctor.sh | tee "$OUT_DIR/doctor.txt"
else
  echo "WARN: scripts/doctor.sh missing" | tee "$OUT_DIR/doctor.txt"
fi

# 2) Create a git archive of the current HEAD (clean snapshot)
git archive --format=tar HEAD | tar -x -C "$OUT_DIR"

# 3) Run any repo-provided PSU demo script from the live repo (not from the archive)
if [ -f run_psu_demo.sh ]; then
  bash ./run_psu_demo.sh | tee "$OUT_DIR/run_psu_demo.txt"
else
  echo "WARN: run_psu_demo.sh missing" | tee "$OUT_DIR/run_psu_demo.txt"
fi

# 4) If you have a bundle contract / verification script, run it
if [ -f scripts/ci_check_overfit3d_contract.sh ]; then
  bash scripts/ci_check_overfit3d_contract.sh | tee "$OUT_DIR/ci_check_overfit3d_contract.txt"
else
  echo "WARN: scripts/ci_check_overfit3d_contract.sh missing" | tee "$OUT_DIR/ci_check_overfit3d_contract.txt"
fi

echo "== PSU_BUNDLE done =="
echo "OUT_DIR=$OUT_DIR"
