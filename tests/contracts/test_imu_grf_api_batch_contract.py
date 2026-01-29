import unittest

import os
import shutil
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.api.imu_to_grf import run_imu_to_grf_batch


class TestIMUGRFAPIBatchContract(unittest.TestCase):
    def test_imu_grf_api_batch_contract_v1(self) -> None:
        fixture = Path("tests/fixtures/imu_sample.csv")
        self.assertTrue(fixture.is_file())

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            shutil.copy(fixture, tmp_path / "a.csv")
            shutil.copy(fixture, tmp_path / "b.csv")

            out = run_imu_to_grf_batch(
                str(tmp_path),
                window_len=256,
                stride=1,
                num_windows=64,
                profile=False,
            )

        # Top-level schema
        self.assertIsInstance(out, dict)
        self.assertEqual(out.get("schema_version"), "imu_grf_batch_v1")

        meta = out.get("metadata", {})
        self.assertIsInstance(meta, dict)
        self.assertEqual(int(meta.get("num_files", 0)), 2)
        self.assertEqual(int(meta.get("num_ok", 0)), 2)
        self.assertEqual(int(meta.get("num_failed", 0)), 0)

        results = out.get("results", [])
        self.assertEqual(len(results), 2)

        for res in results:
            self.assertTrue(res.get("ok"), msg=f"unexpected failure result: {res}")
            self.assertIn("output", res)
            output = res["output"]
            self.assertIsInstance(output, dict)
            self.assertEqual(output.get("shape"), [64, 256, 1])
            # With profile=False, no per-result perf block should be present.
            self.assertNotIn("perf", res)

        summary = out.get("summary", {})
        self.assertIsInstance(summary, dict)
        self.assertGreaterEqual(float(summary.get("ok_rate", 0.0)), 1.0)
        # With profile=False, aggregate perf keys should be absent.
        self.assertNotIn("p50_total_ms", summary)
        self.assertNotIn("p95_total_ms", summary)
        self.assertNotIn("max_rss_mb", summary)


if __name__ == "__main__":
    unittest.main()
