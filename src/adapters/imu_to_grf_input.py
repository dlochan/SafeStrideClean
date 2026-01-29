from __future__ import annotations

from pathlib import Path

import numpy as np

from src.imu.schema import parse_imu_csv
from src.imu.windowing import make_sliding_windows
from src.vnext.data.imu_schema import get_feature_columns


def build_grf_input_from_imu_csv(
    csv_path: Path,
    *,
    window_len: int = 256,
    stride: int | None = None,
    num_windows: int = 1,
) -> np.ndarray:
    """Build a GRF-model input tensor from a dual-IMU CSV.

    Expected model input shape (per src/vnext/models/vnext_fz.py and vnext_grf3d.py):
      x: (B, T, C) where:
        - B = number of windows
        - T = window_len
        - C = len(get_feature_columns()) channels in canonical order
    """

    csv_path = Path(csv_path)
    if window_len <= 0:
        raise ValueError("window_len must be > 0")
    if num_windows <= 0:
        raise ValueError("num_windows must be > 0")
    if stride is None:
        stride = int(window_len)

    rows = parse_imu_csv(csv_path)

    # Build an aligned sequence using the existing windowing code.
    # We use window_len=1 to extract the aligned per-timestep tensor even when
    # the fixture is shorter than the model's window length.
    aligned_steps = make_sliding_windows(
        rows,
        window_len=1,
        stride=1,
        sensor_order=["thigh", "shank"],
    )

    expected = get_feature_columns()
    # vNext canonical columns use axx/axy/axz and gxx/gxy/gxz naming, while our
    # IMU utilities produce ax/ay/az and gx/gy/gz. Map deterministically.
    idx_by_name = {n: i for i, n in enumerate(aligned_steps.channel_names)}
    reorder_idx: list[int] = []
    for name in expected:
        if "_" not in name:
            raise ValueError(f"Unexpected canonical feature column: {name}")
        prefix, tag = name.split("_", 1)
        if prefix.startswith("ax") and len(prefix) >= 3:
            axis = prefix[2]
            src = f"a{axis}_{tag}"
        elif prefix.startswith("gx") and len(prefix) >= 3:
            axis = prefix[2]
            src = f"g{axis}_{tag}"
        else:
            raise ValueError(f"Unexpected canonical feature column: {name}")
        if src not in idx_by_name:
            raise ValueError(
                "IMU channel mapping failed; required source channel missing. "
                f"canonical={name} expected_source={src} available={aligned_steps.channel_names}"
            )
        reorder_idx.append(idx_by_name[src])

    # aligned_steps.windows: (N, C_src, 1) -> seq: (T_aligned, C_src)
    X_steps = np.asarray(aligned_steps.windows, dtype=np.float32)
    seq_src_tc = X_steps[:, :, 0]
    seq_tc = seq_src_tc[:, reorder_idx]  # (T_aligned, C_expected)

    # Build windows deterministically from the aligned sequence.
    # If the sequence is too short for even one full window, pad by repeating
    # the final sample. If there are too few windows, pad by repeating the
    # final window. If there are too many, truncate.
    if seq_tc.shape[0] < window_len:
        pad = window_len - seq_tc.shape[0]
        last = seq_tc[-1:, :]
        seq_tc = np.concatenate([seq_tc, np.repeat(last, pad, axis=0)], axis=0)

    windows = []
    for start in range(0, seq_tc.shape[0] - window_len + 1, int(stride)):
        windows.append(seq_tc[start : start + window_len, :])
        if len(windows) >= int(num_windows):
            break

    if not windows:
        raise ValueError("Failed to construct any windows")

    X_btc = np.stack(windows, axis=0).astype(np.float32, copy=False)  # (B,T,C)
    if X_btc.shape[0] < int(num_windows):
        last_w = X_btc[-1:, :, :]
        pad_w = int(num_windows) - X_btc.shape[0]
        X_btc = np.concatenate([X_btc, np.repeat(last_w, pad_w, axis=0)], axis=0)
    elif X_btc.shape[0] > int(num_windows):
        X_btc = X_btc[: int(num_windows), :, :]

    if not np.isfinite(X_btc).all():
        raise ValueError("Non-finite values in GRF input tensor")
    return X_btc
