"""Public API package for SafeStride IMU→GRF inference.

This package exposes a stable, JSON-serializable entrypoint `run_imu_to_grf`
that wraps the internal IMU→GRF adapter and deterministic inference path.
"""

from .imu_to_grf import run_imu_to_grf

__all__ = ["run_imu_to_grf"]
