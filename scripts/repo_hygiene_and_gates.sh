#!/usr/bin/env bash
set -euo pipefail

__SKIP_GATES__=0
if [ "${1:-}" = "--skip-gates" ]; then
  __SKIP_GATES__=1
  shift || true
fi

pass_count=0
fail_count=0

pass() {
  echo "PASS: $*"
  pass_count=$((pass_count + 1))
}

fail() {
  echo "FAIL: $*" >&2
  fail_count=$((fail_count + 1))
}

note() {
  echo "NOTE: $*" >&2
}

for pollution_dir in app external reports; do
  if [ -d "${pollution_dir}" ]; then
    untracked_in_dir="$(git ls-files --others --exclude-standard -- "${pollution_dir}" || true)"
    if [ -n "${untracked_in_dir}" ]; then
      fail "POLLUTION PREFLIGHT: untracked files detected under '${pollution_dir}'"
      echo "--- first 50 untracked paths under ${pollution_dir} ---" >&2
      echo "${untracked_in_dir}" | sed -n '1,50p' >&2
      echo "Delete the accidental folder(s) or explicitly add legitimate sources to git, then re-run." >&2
      exit 4
    fi
    note "POLLUTION PREFLIGHT: '${pollution_dir}' exists but has no untracked (non-ignored) files"
  fi
done

status_out="$(git status --porcelain=v1)"
if [ -n "${status_out}" ]; then
  fail "working tree not clean"
  echo "--- git status --porcelain=v1 ---" >&2
  echo "${status_out}" >&2
  exit 2
fi
pass "working tree clean"

if git check-ignore -v artifacts/ >/dev/null 2>&1; then
  pass "artifacts/ is ignored"
else
  fail "artifacts/ is NOT ignored"
  echo "--- git check-ignore -v artifacts/ ---" >&2
  git check-ignore -v artifacts/ >&2 || true
  exit 3
fi

if git check-ignore -v .DS_Store >/dev/null 2>&1; then
  note ".DS_Store would be ignored"
else
  note ".DS_Store does not appear ignored (non-fatal)"
fi

if git check-ignore -v ._README.md >/dev/null 2>&1; then
  note "AppleDouble sidecars (._*) would be ignored"
else
  note "AppleDouble sidecars (._*) do not appear ignored (non-fatal)"
fi

if git check-ignore -v app/__pycache__/x.pyc >/dev/null 2>&1; then
  note "python bytecode caches (__pycache__/ *.pyc) would be ignored"
else
  note "python bytecode caches (__pycache__/ *.pyc) do not appear ignored (non-fatal)"
fi

if [ "$__SKIP_GATES__" -eq 1 ]; then
  echo "NOTE: gates skipped (--skip-gates)"
else
if [ "${__SKIP_GATES__:-0}" -eq 1 ]; then
  echo "NOTE: gates skipped (--skip-gates)"
else
  echo "--- running 3D smoke ---"
  bash scripts/mp_converge_3d_smoke.sh
  echo "PASS: mp_converge_3d_smoke"
  echo "--- running CI overfit3d contract ---"
  bash scripts/ci_check_overfit3d_contract.sh
  echo "PASS: ci_check_overfit3d_contract"
fi

echo "--- summary ---"
echo "PASS_COUNT=${pass_count}"
echo "FAIL_COUNT=${fail_count}"
if [ "${fail_count}" -eq 0 ]; then
  echo "PASS: repo hygiene + gates"
  exit 0
fi

echo "FAIL: repo hygiene + gates" >&2
exit 1
