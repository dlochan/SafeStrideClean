import unittest
from pathlib import Path

import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.imu.schema import parse_imu_csv


class TestIMUSchema(unittest.TestCase):
    def test_parse_fixture(self):
        p = Path("tests/fixtures/imu_sample.csv")
        rows = parse_imu_csv(p)
        self.assertGreaterEqual(len(rows), 2)

        sensors = sorted({r.sensor_id for r in rows})
        self.assertEqual(sensors, ["shank", "thigh"])

        self.assertTrue(all(r.mag_x is not None for r in rows))

        # Expect at least one interleaving between streams in the fixture.
        self.assertGreaterEqual(len(rows), 4)
        self.assertNotEqual(rows[0].sensor_id, rows[1].sensor_id)

        by_sensor = {}
        for r in rows:
            by_sensor.setdefault(r.sensor_id, []).append(r.t_ms)

        for sensor_id, t in sorted(by_sensor.items()):
            for a, b in zip(t, t[1:]):
                self.assertLessEqual(a, b, msg=f"non_monotonic_t_ms sensor={sensor_id} a={a} b={b}")


if __name__ == "__main__":
    unittest.main()
