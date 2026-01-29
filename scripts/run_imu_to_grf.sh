#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 PATH_TO_IMU_CSV" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CSV="$1"

python3 - "$CSV" << 'PY'
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from src.api import run_imu_to_grf


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Expected CSV path as argv[1]")

    csv_path = sys.argv[1]
    result = run_imu_to_grf(csv_path, window_len=256, stride=1, num_windows=64)

    # Always print JSON to stdout
    print(json.dumps(result, indent=2, sort_keys=True))

    out_json = os.environ.get("OUT_JSON")
    if out_json:
        p = Path(out_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
PY
