#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
python3 - << 'PY'
import sys
from pathlib import Path

repo = Path.cwd()
sys.path.insert(0, str(repo))

from src.imu.schema import parse_imu_csv
from src.imu.windowing import make_sliding_windows
from src.imu.features import extract_features

rows = parse_imu_csv(repo / "tests/fixtures/imu_sample.csv")
windowed = make_sliding_windows(rows, window_len=3, stride=1)
feats = extract_features(windowed, include_magnitude=False)

print(f"IMU_FEATURES_TENSOR_SHAPE={feats.X.shape}")
PY
