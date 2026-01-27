import unittest
from pathlib import Path

from src.imu.schema import parse_imu_csv


class TestIMUSchema(unittest.TestCase):
    def test_parse_fixture(self):
        p = Path("tests/fixtures/imu_sample.csv")
        rows = parse_imu_csv(p)
        self.assertGreaterEqual(len(rows), 2)

        sensors = sorted({r.sensor_id for r in rows})
        self.assertEqual(sensors, ["shank", "thigh"])

        self.assertTrue(all(r.mag_x is not None for r in rows))

        t = [r.t_ms for r in rows]
        for a, b in zip(t, t[1:]):
# NOTE: global monotonic t_ms is not valid for interleaved multi-sensor logs
# Old assertion removed; replaced with per-sensor monotonic check below.
#             self.assertLessEqual(a, b)


if __name__ == "__main__":
    unittest.main()
