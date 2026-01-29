import unittest

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.adapters.imu_ingest import ingest_imu_csv


class TestIMUIngestAdapter(unittest.TestCase):
    def test_ingest_fixture(self):
        out = ingest_imu_csv(Path("tests/fixtures/imu_sample.csv"), window_len=3, stride=1)

        self.assertIn("meta", out)
        self.assertIn("X", out)
        self.assertIn("t0_ms", out)
        self.assertIn("sensor_map", out)

        X = out["X"]
        self.assertEqual(X.ndim, 3)
        self.assertGreater(X.shape[0], 0)
        self.assertGreater(X.shape[1], 0)
        self.assertGreater(X.shape[2], 0)
        self.assertTrue(np.isfinite(X).all())

        meta = out["meta"]
        self.assertEqual(meta["window_len"], 3)
        self.assertEqual(meta["stride"], 1)
        self.assertEqual(meta["num_windows"], X.shape[0])
        self.assertEqual(meta["num_sensors"], len(meta["sensor_ids"]))


if __name__ == "__main__":
    unittest.main()
