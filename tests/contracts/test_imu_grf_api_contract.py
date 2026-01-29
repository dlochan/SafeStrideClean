import unittest

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.api import run_imu_to_grf
from src.vnext.data.imu_schema import get_feature_columns


class TestIMUGRFAPIContract(unittest.TestCase):
    def test_imu_grf_api_contract_v1(self) -> None:
        out = run_imu_to_grf(
            "tests/fixtures/imu_sample.csv",
            window_len=256,
            stride=1,
            num_windows=64,
        )

        # Top-level schema checks
        self.assertIsInstance(out, dict)
        self.assertEqual(out.get("schema_version"), "imu_grf_v1")
        self.assertEqual(out.get("model"), "vnext_fz")

        feature_cols = get_feature_columns()

        # Input block checks
        inp = out.get("input", {})
        self.assertIsInstance(inp, dict)
        self.assertEqual(inp.get("channels"), len(feature_cols))

        # Output shape and units
        output = out.get("output", {})
        self.assertIsInstance(output, dict)
        self.assertEqual(output.get("shape"), [64, 256, 1])
        self.assertEqual(output.get("units"), "newtons")

        # Stats must exist and be finite, with finite_fraction == 1.0
        stats = output.get("stats")
        self.assertIsInstance(stats, dict)
        for key in ("min", "max", "mean", "std", "finite_fraction"):
            self.assertIn(key, stats)

        for key in ("min", "max", "mean", "std"):
            val = float(stats[key])
            self.assertTrue(np.isfinite(val), f"stats[{key}] is not finite: {val}")

        self.assertEqual(float(stats["finite_fraction"]), 1.0)


if __name__ == "__main__":
    unittest.main()
