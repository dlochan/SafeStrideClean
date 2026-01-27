#!/usr/bin/env bash
set -euo pipefail

msg="${1:-save}"
branch="$(git branch --show-current)"

# Run gates first so you never push broken main by accident
bash scripts/repo_hygiene_and_gates.sh

# Commit if needed
if [ -n "$(git status --porcelain=v1)" ]; then
  git add -A
  git commit -m "$msg"
else
  echo "NOTE: nothing to commit"
fi

# Push only if upstream exists, otherwise set it
if git rev-parse --abbrev-ref --symbolic-full-name '@{u}' >/dev/null 2>&1; then
  git push
else
  git push -u origin "$branch"
fi
