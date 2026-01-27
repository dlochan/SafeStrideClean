#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/mega_loop.sh
#   bash scripts/mega_loop.sh --max-iter 5
#   bash scripts/mega_loop.sh --no-push
#   bash scripts/mega_loop.sh --commit-msg "checkpoint: loop pass"

MAX_ITER=6
DO_PUSH=1
COMMIT_MSG="checkpoint: mega loop"

while [ $# -gt 0 ]; do
  case "$1" in
    --max-iter) MAX_ITER="$2"; shift 2 ;;
    --no-push) DO_PUSH=0; shift ;;
    --commit-msg) COMMIT_MSG="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 2 ;;
  esac
done

ts="$(date +%Y%m%d-%H%M%S)"
LOG="artifacts/mega_loop_${ts}.log"
: > "$LOG"

say() { echo "$*" | tee -a "$LOG"; }

say "## START ts=${ts}"
say "## PWD=$(pwd)"
say "## HEAD=$(git rev-parse --short HEAD)"
say "## BRANCH=$(git branch --show-current)"

iter=1
while [ "$iter" -le "$MAX_ITER" ]; do
  say ""
  say "## ITER=$iter/$MAX_ITER"

  # Always start clean: show status, but do not auto-delete anything.
  say "## STATUS_PORCELAIN_BEFORE"
  git status --porcelain=v1 | tee -a "$LOG"

  # 1) Hard gates (your repo_hygiene_and_gates.sh already runs smoke + CI contract + non-regression)
  say "## RUN repo_hygiene_and_gates.sh"
  if bash scripts/repo_hygiene_and_gates.sh 2>&1 | tee -a "$LOG"; then
    say "## PASS gates"
  else
    rc="${PIPESTATUS[0]:-1}"
    say "## FAIL gates rc=$rc"
    say "## TRIAGE"
    tail -n 120 "$LOG" | sed -n '1,120p' | tee -a "$LOG" >/dev/null || true
    say "## STOP (fix required). Log=$LOG"
    exit "$rc"
  fi

  # 2) If pytest exists, run it (fast). If not, skip.
  if [ -d tests ] || compgen -G "test_*.py" >/dev/null; then
    say "## RUN pytest (if available)"
    if python3 -m pytest -q 2>/dev/null | tee -a "$LOG"; then
      say "## PASS pytest"
    else
      say "## NOTE pytest failed or not installed"
      say "## STOP. Log=$LOG"
      exit 30
    fi
  else
    say "## NOTE no tests detected (pytest skipped)"
  fi

  # 3) Optional: run any repo-local “doctor” if present
  if [ -f scripts/doctor.sh ]; then
    say "## RUN doctor.sh"
    bash scripts/doctor.sh 2>&1 | tee -a "$LOG" || true
  fi

  # 4) Save checkpoint (commit + optional push) only if changed
  if [ -n "$(git status --porcelain=v1)" ]; then
    say "## CHANGES detected: committing"
    git add -A
    git commit -m "$COMMIT_MSG (iter $iter)" | tee -a "$LOG"
    if [ "$DO_PUSH" -eq 1 ]; then
      say "## PUSH"
      git push 2>&1 | tee -a "$LOG" || { say "## NOTE push failed (network/auth). Continuing."; }
    else
      say "## NOTE push disabled"
    fi
  else
    say "## NOTE no changes to commit"
  fi

  say "## ITER_DONE iter=$iter"
  iter=$((iter + 1))
done

say ""
say "## DONE all iterations completed cleanly"
say "## LOG=$LOG"
