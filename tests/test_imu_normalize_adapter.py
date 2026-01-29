import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.adapters.imu_normalize import (
    NormalizationDebug,
    normalize_imu_csv_to_canon_df,
    normalize_imu_csv_to_canon_df_with_debug,
)
from src.adapters.imu_to_grf_input import build_grf_input_from_imu_csv
from src.vnext.data.imu_schema import get_feature_columns


class TestImuNormalizeAdapter(unittest.TestCase):
    def test_normalize_fixture_columns_and_dtypes(self) -> None:
        csv_path = Path("tests/fixtures/imu_messy.csv")
        df = normalize_imu_csv_to_canon_df(str(csv_path))

        feature_cols = get_feature_columns()
        self.assertEqual(list(df.columns), feature_cols)

        values = df.to_numpy(dtype=np.float32, copy=False)
        self.assertEqual(values.dtype, np.float32)
        self.assertTrue(np.isfinite(values).all())

    def test_end_to_end_normalize_then_build_input(self) -> None:
        csv_path = Path("tests/fixtures/imu_messy.csv")
        df = normalize_imu_csv_to_canon_df(str(csv_path))

        feature_cols = get_feature_columns()
        tags = sorted({name.split("_", 1)[1] for name in feature_cols})

        rows = []
        for i in range(len(df)):
            t_ms = int(i * 10)
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

    def test_alias_resolution_and_duplicate_prefer_first(self) -> None:
        csv_path = Path("tests/fixtures/imu_messy.csv")

        raw_df = pd.read_csv(csv_path)
        df, debug = normalize_imu_csv_to_canon_df_with_debug(str(csv_path))

        feature_cols = get_feature_columns()
        self.assertEqual(list(df.columns), feature_cols)

        # Thigh X should come from the primary alias column, not the secondary
        primary_alias = "Ax Thigh"
        secondary_alias = "Accel_X thigh"
        canon_col = "axx_thigh"

        self.assertIn(primary_alias, raw_df.columns)
        self.assertIn(secondary_alias, raw_df.columns)
        self.assertIn(canon_col, df.columns)

        raw_primary = raw_df[primary_alias].to_numpy(dtype=np.float32)
        raw_secondary = raw_df[secondary_alias].to_numpy(dtype=np.float32)
        canon_vals = df[canon_col].to_numpy(dtype=np.float32)

        # The canonical channel should match the primary alias exactly
        self.assertTrue(np.allclose(canon_vals, raw_primary))
        # And differ from the secondary alias (so we know which one was chosen)
        self.assertFalse(np.allclose(canon_vals, raw_secondary))

        # Debug payload should record both aliases being used for the same canon
        self.assertIsInstance(debug, NormalizationDebug)
        self.assertIn((primary_alias, canon_col), debug.used_aliases)
        self.assertIn((secondary_alias, canon_col), debug.used_aliases)

    def test_missing_canon_hard_fails(self) -> None:
        # Build a tiny CSV that is missing exactly one canonical feature
        # (gxz_shank) while providing aliases for all others.
        feature_cols = get_feature_columns()
        self.assertIn("gxz_shank", feature_cols)

        csv_text = """time_ms,Ax Thigh,Ay-Thigh,Az thigh   ,Gx Thigh,GYRO_Y thigh   ,gyro_z_thigh,ax_shank,accel_y_shank,Az-Shank,Gx Shank,gyro_y_shank
0,0.10,0.20,0.30,0.01,0.02,0.03,0.40,0.50,0.60,0.04,0.05
10,0.11,0.21,0.31,0.011,0.021,0.031,0.41,0.51,0.61,0.041,0.051
"""

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp:
            tmp.write(csv_text)
            tmp_path = tmp.name

        try:
            with self.assertRaises(ValueError) as ctx:
                normalize_imu_csv_to_canon_df(tmp_path)
            msg = str(ctx.exception)
            self.assertIn("gxz_shank", msg)
        finally:
            os.remove(tmp_path)


if __name__ == "__main__":
    unittest.main()
