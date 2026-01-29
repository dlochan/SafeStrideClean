import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.adapters.imu_normalize import normalize_imu_csv
from src.adapters.imu_to_grf_input import build_grf_input_from_imu_csv
from src.vnext.data.imu_schema import get_feature_columns


class TestImuNormalizeAdapter(unittest.TestCase):
    def test_normalize_fixture_columns_and_dtypes(self) -> None:
        csv_path = Path("tests/fixtures/imu_messy.csv")
        df = normalize_imu_csv(str(csv_path))

        feature_cols = get_feature_columns()
        expected_cols = ["time_s"] + feature_cols
        self.assertEqual(list(df.columns), expected_cols)

        values = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
        self.assertEqual(values.dtype, np.float32)
        self.assertTrue(np.isfinite(values).all())

    def test_end_to_end_normalize_then_build_input(self) -> None:
        csv_path = Path("tests/fixtures/imu_messy.csv")
        df = normalize_imu_csv(str(csv_path))

        feature_cols = get_feature_columns()
        tags = sorted({name.split("_", 1)[1] for name in feature_cols})

        rows = []
        for i in range(len(df)):
            t_s = float(df.iloc[i]["time_s"])
            t_ms = int(round(t_s * 1000.0))
            for tag in tags:
                vals = {}
                for name in feature_cols:
                    prefix, name_tag = name.split("_", 1)
                    if name_tag != tag:
                        continue
                    kind = prefix[0]
                    axis = prefix[-1]
                    if kind not in ("a", "g") or axis not in ("x", "y", "z"):
                        continue
                    key = f"{'a' if kind == 'a' else 'g'}{axis}"
                    vals[key] = float(df.iloc[i][name])
                rows.append(
                    {
                        "t_ms": t_ms,
                        "sensor_id": tag,
                        "ax": vals["ax"],
                        "ay": vals["ay"],
                        "az": vals["az"],
                        "gx": vals["gx"],
                        "gy": vals["gy"],
                        "gz": vals["gz"],
                    }
                )

        adapter_df = pd.DataFrame(
            rows,
            columns=["t_ms", "sensor_id", "ax", "ay", "az", "gx", "gy", "gz"],
        )
        out_dir = Path("artifacts")
        out_dir.mkdir(parents=True, exist_ok=True)
        adapter_csv = out_dir / "imu_normalize_adapter_tmp.csv"
        adapter_df.to_csv(adapter_csv, index=False)

        X = build_grf_input_from_imu_csv(
            adapter_csv, window_len=256, stride=1, num_windows=64
        )

        self.assertEqual(X.dtype, np.float32)
        self.assertTrue(np.isfinite(X).all())

        self.assertEqual(X.shape[0], 64)
        self.assertEqual(X.shape[1], 256)
        self.assertEqual(X.shape[2], len(feature_cols))


if __name__ == "__main__":
    unittest.main()
