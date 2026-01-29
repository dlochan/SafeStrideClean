import unittest

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.adapters.imu_to_grf_input import build_grf_input_from_imu_csv


class TestImuToGrfInputAdapter(unittest.TestCase):
    def test_shape_dtype_finite_deterministic(self) -> None:
        csv_path = Path("tests/fixtures/imu_sample.csv")

        X1_single = build_grf_input_from_imu_csv(csv_path, window_len=256, stride=256, num_windows=1)
        X2_single = build_grf_input_from_imu_csv(csv_path, window_len=256, stride=256, num_windows=1)

        self.assertEqual(X1_single.shape[0], 1)
        self.assertEqual(X1_single.shape[1], 256)
        self.assertGreater(X1_single.shape[2], 0)
        self.assertEqual(X1_single.dtype, np.float32)
        self.assertTrue(np.isfinite(X1_single).all())
        self.assertTrue(np.array_equal(X1_single, X2_single))

        X1_batch = build_grf_input_from_imu_csv(csv_path, window_len=256, stride=4, num_windows=64)
        X2_batch = build_grf_input_from_imu_csv(csv_path, window_len=256, stride=4, num_windows=64)

        self.assertEqual(X1_batch.shape[0], 64)
        self.assertEqual(X1_batch.shape[1], 256)
        self.assertEqual(X1_batch.shape[2], X1_single.shape[2])
        self.assertEqual(X1_batch.dtype, np.float32)
        self.assertTrue(np.isfinite(X1_batch).all())
        self.assertTrue(np.array_equal(X1_batch, X2_batch))


if __name__ == "__main__":
    unittest.main()
