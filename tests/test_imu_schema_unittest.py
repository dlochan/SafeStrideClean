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

        by_sensor = {}
        for r in rows:
            by_sensor.setdefault(r.sensor_id, []).append(r.t_ms)

        for sensor_id, t in sorted(by_sensor.items()):
            for a, b in zip(t, t[1:]):
                self.assertLessEqual(a, b, msg=f"non_monotonic_t_ms sensor={sensor_id} a={a} b={b}")


if __name__ == "__main__":
    unittest.main()
