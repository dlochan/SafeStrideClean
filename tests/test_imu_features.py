import unittest

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.imu.schema import IMURow
from src.imu.windowing import make_sliding_windows
from src.imu.features import extract_features


class TestIMUFeatures(unittest.TestCase):
    def test_extract_features_shape(self):
        rows = [
            IMURow(t_ms=0, sensor_id="thigh", ax=0.0, ay=0.0, az=0.0, gx=0.0, gy=0.0, gz=0.0),
            IMURow(t_ms=0, sensor_id="shank", ax=1.0, ay=1.0, az=1.0, gx=1.0, gy=1.0, gz=1.0),
            IMURow(t_ms=10, sensor_id="thigh", ax=0.1, ay=0.1, az=0.1, gx=0.1, gy=0.1, gz=0.1),
            IMURow(t_ms=10, sensor_id="shank", ax=1.1, ay=1.1, az=1.1, gx=1.1, gy=1.1, gz=1.1),
            IMURow(t_ms=20, sensor_id="thigh", ax=0.2, ay=0.2, az=0.2, gx=0.2, gy=0.2, gz=0.2),
            IMURow(t_ms=20, sensor_id="shank", ax=1.2, ay=1.2, az=1.2, gx=1.2, gy=1.2, gz=1.2),
        ]

        windowed = make_sliding_windows(rows, window_len=3, stride=1)
        feats = extract_features(windowed, include_magnitude=True)

        self.assertEqual(windowed.windows.shape, (1, 12, 3))
        self.assertEqual(feats.X.shape[0], 1)
        self.assertEqual(feats.X.shape[2], 3)
        self.assertTrue(np.isfinite(feats.X).all())
        self.assertGreaterEqual(feats.X.shape[1], 12)  # at least raw channels


if __name__ == "__main__":
    unittest.main()
