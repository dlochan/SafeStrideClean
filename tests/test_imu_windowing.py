import unittest

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.imu.schema import IMURow
from src.imu.windowing import make_sliding_windows


class TestIMUWindowing(unittest.TestCase):
    def test_make_sliding_windows_two_sensors(self):
        rows = [
            IMURow(t_ms=0, sensor_id="thigh", ax=0.0, ay=0.0, az=0.0, gx=0.0, gy=0.0, gz=0.0),
            IMURow(t_ms=0, sensor_id="shank", ax=1.0, ay=1.0, az=1.0, gx=1.0, gy=1.0, gz=1.0),
            IMURow(t_ms=10, sensor_id="thigh", ax=0.1, ay=0.1, az=0.1, gx=0.1, gy=0.1, gz=0.1),
            IMURow(t_ms=10, sensor_id="shank", ax=1.1, ay=1.1, az=1.1, gx=1.1, gy=1.1, gz=1.1),
            IMURow(t_ms=20, sensor_id="thigh", ax=0.2, ay=0.2, az=0.2, gx=0.2, gy=0.2, gz=0.2),
            IMURow(t_ms=20, sensor_id="shank", ax=1.2, ay=1.2, az=1.2, gx=1.2, gy=1.2, gz=1.2),
            IMURow(t_ms=30, sensor_id="thigh", ax=0.3, ay=0.3, az=0.3, gx=0.3, gy=0.3, gz=0.3),
            IMURow(t_ms=30, sensor_id="shank", ax=1.3, ay=1.3, az=1.3, gx=1.3, gy=1.3, gz=1.3),
        ]

        out = make_sliding_windows(rows, window_len=3, stride=1)
        self.assertEqual(out.windows.shape, (2, 12, 3))
        self.assertEqual(list(out.t0_ms), [0, 10])
        self.assertEqual(len(out.channel_names), 12)
        self.assertEqual(out.channel_names[0], "ax_thigh")
        self.assertEqual(out.channel_names[6], "ax_shank")


if __name__ == "__main__":
    unittest.main()
