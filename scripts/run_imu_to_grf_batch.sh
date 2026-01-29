#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 PATH_TO_IMU_DIR" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMU_DIR="$1"

python3 - "$IMU_DIR" << 'PY'
from __future__ import annotations

import json
import sys

from src.api.imu_to_grf import run_imu_to_grf_batch


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Expected IMU directory path as argv[1]")

    imu_dir = sys.argv[1]
    result = run_imu_to_grf_batch(
        imu_dir,
        window_len=256,
        stride=1,
        num_windows=64,
        profile=True,
    )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
PY
