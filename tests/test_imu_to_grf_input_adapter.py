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

        X1 = build_grf_input_from_imu_csv(csv_path, window_len=3, stride=3)
        X2 = build_grf_input_from_imu_csv(csv_path, window_len=3, stride=3)

        self.assertEqual(X1.shape, (1, 3, 12))
        self.assertEqual(X1.dtype, np.float32)
        self.assertTrue(np.isfinite(X1).all())

        self.assertTrue(np.array_equal(X1, X2))


if __name__ == "__main__":
    unittest.main()
