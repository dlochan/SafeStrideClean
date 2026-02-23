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
  pass_count=$((pass_count+1))
  echo "PASS: $*"
}

fail() {
  fail_count=$((fail_count+1))
  echo "FAIL: $*" >&2
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

  echo "--- running IMU schema smoke ---"
  if bash scripts/imu_schema_smoke.sh; then
    pass "imu_schema_smoke"
  else
    fail "imu_schema_smoke"
    exit 12
  fi

  echo "--- running IMU features smoke ---"
  if bash scripts/imu_features_smoke.sh; then
    pass "imu_features_smoke"
  else
    fail "imu_features_smoke"
    exit 12
  fi

  echo "--- running IMU contract ---"
  if bash scripts/ci_check_imu_contract.sh; then
    pass "ci_check_imu_contract"
  else
    fail "ci_check_imu_contract"
    exit 13
  fi

  echo "--- running IMU ingest smoke ---"
  if bash scripts/imu_ingest_smoke.sh; then
    pass "imu_ingest_smoke"
  else
    fail "imu_ingest_smoke"
    exit 14
  fi

  echo "--- running IMU normalize smoke ---"
  if bash scripts/imu_normalize_smoke.sh; then
    pass "imu_normalize_smoke"
  else
    fail "imu_normalize_smoke"
    exit 34
  fi

  echo "--- running IMU normalize contract ---"
  if bash scripts/ci_check_imu_normalize_contract.sh; then
    pass "ci_check_imu_normalize_contract"
  else
    fail "ci_check_imu_normalize_contract"
    exit 44
  fi

  echo "--- running IMU -> GRF input smoke ---"
  if bash scripts/imu_to_grf_input_smoke.sh; then
    pass "imu_to_grf_input_smoke"
  else
    fail "imu_to_grf_input_smoke"
    exit 15
  fi

  echo "--- running IMU -> GRF inference smoke ---"
  if bash scripts/imu_to_grf_infer_smoke.sh; then
    pass "imu_to_grf_infer_smoke"
  else
    fail "imu_to_grf_infer_smoke"
    exit 16
  fi

  echo "--- running IMU -> GRF inference non-regression contract ---"
  if bash scripts/ci_check_imu_to_grf_contract.sh; then
    pass "ci_check_imu_to_grf_contract"
  else
    fail "ci_check_imu_to_grf_contract"
    exit 28
  fi

  echo "--- running knee moment 2D non-regression contract ---"
  if bash scripts/ci_check_knee_moment_2d_contract.sh; then
    pass "ci_check_knee_moment_2d_contract"
  else
    fail "ci_check_knee_moment_2d_contract"
    exit 52
  fi

  echo "--- running knee metrics 2D non-regression contract ---"
  if bash scripts/ci_check_knee_metrics_2d_contract.sh; then
    pass "ci_check_knee_metrics_2d_contract"
  else
    fail "ci_check_knee_metrics_2d_contract"
    exit 53
  fi

  echo "--- running IMU -> GRF inference contract ---"
  if bash scripts/ci_check_imu_infer_contract.sh; then
    pass "ci_check_imu_infer_contract"
  else
    fail "ci_check_imu_infer_contract"
    exit 18
  fi

  echo "--- running IMU -> GRF API contract ---"
  if python3 -m unittest -q tests.contracts.test_imu_grf_api_contract; then
    pass "imu_grf_api_contract"
  else
    fail "imu_grf_api_contract"
    exit 20
  fi

  echo "--- running IMU -> GRF API batch contract ---"
  if bash scripts/ci_check_imu_grf_api_batch_contract.sh; then
    pass "ci_check_imu_grf_api_batch_contract"
  else
    fail "ci_check_imu_grf_api_batch_contract"
    exit 26
  fi

  echo "--- running IMU→GRF perf gate ---"
  if bash scripts/ci_check_imu_grf_perf.sh; then
    pass "ci_check_imu_grf_perf"
  else
    fail "ci_check_imu_grf_perf"
    exit 22
  fi

  echo "--- running PSU bundle check ---"
  if bash scripts/ci_check_psu_bundle.sh; then
    pass "ci_check_psu_bundle"
  else
    fail "ci_check_psu_bundle"
    exit 24
  fi

  echo "--- running PSU bundle contract ---"
  if bash scripts/ci_check_psu_bundle_contract.sh; then
    pass "ci_check_psu_bundle_contract"
  else
    fail "ci_check_psu_bundle_contract"
    exit 30
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
