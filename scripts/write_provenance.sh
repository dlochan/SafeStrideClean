#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

# UTC timestamp in ISO-8601
printf 'timestamp_utc=%s\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

# Git metadata
printf 'git_commit_short=%s\n' "$(git rev-parse --short HEAD)"
printf 'git_commit_full=%s\n' "$(git rev-parse HEAD)"
printf 'git_branch=%s\n' "$(git branch --show-current)"

# System info
printf 'uname=%s\n' "$(uname -a)"

# Python version (best effort)
python3_version="$(python3 --version 2>/dev/null || true)"
if [ -n "$python3_version" ]; then
  printf 'python3_version=%s\n' "$python3_version"
else
  printf 'python3_version=unavailable\n'
fi

# pip freeze line count (best effort, do not fail if pip missing)
if command -v pip3 >/dev/null 2>&1; then
  freeze_count="$(pip3 freeze 2>/dev/null | wc -l | tr -d ' ')"
  printf 'pip3_freeze_line_count=%s\n' "$freeze_count"
else
  printf 'pip3_freeze_line_count=unknown\n'
fi

# Torch version (best effort)
python3 - << 'PY'
from __future__ import annotations

try:
    import torch  # type: ignore
    v = getattr(torch, "__version__", "unknown")
    print(f"torch_version={v}")
except Exception:
    print("torch_version=unavailable")
PY
