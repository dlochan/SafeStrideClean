from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch


@dataclass
class KinematicFeatureConfig:
    enable_kinematics: bool = True


@dataclass
class KinematicFeatureBuilder:
    """Builds optional kinematic feature augmentation for dual-IMU windows.

    Takes canonical IMU feature names and per-sensor slices and can transform an
    (B, T, C_in) tensor into (B, T, C_out) with extra channels.
    """

    in_feature_names: List[str]
    sensor_slices: Dict[str, slice]
    cfg: KinematicFeatureConfig
    out_feature_names: List[str] = field(init=False)

    def __post_init__(self) -> None:
        # If features are disabled, out_feature_names == in_feature_names
        if not self.cfg.enable_kinematics:
            self.out_feature_names = list(self.in_feature_names)
            return

        # When enabled, we append eight derived channels in a fixed order.
        self.out_feature_names = list(self.in_feature_names)
        self.out_feature_names.extend(
            [
                "a_mag_thigh",
                "a_mag_shank",
                "g_mag_thigh",
                "g_mag_shank",
                "a_mag_diff_thigh_shank",
                "g_mag_diff_thigh_shank",
                "a_mag_thigh_dt",
                "a_mag_shank_dt",
            ]
        )

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature augmentation.

        x: (T, C) or (B, T, C).
        Returns tensor with same leading dims and possibly more channels.
        If cfg.enable_kinematics is False, returns x unchanged.
        """
        if not self.cfg.enable_kinematics:
            return x

        if x.dim() == 2:
            # (T, C) -> add a dummy batch dim for unified logic
            x_in = x.unsqueeze(0)
            squeeze_batch = True
        elif x.dim() == 3:
            x_in = x
            squeeze_batch = False
        else:
            raise ValueError(f"Expected x to have 2 or 3 dims, got shape {tuple(x.shape)}")

        # x_in: (B, T, C_in)
        B, T, C = x_in.shape

        thigh_slice = self.sensor_slices.get("thigh")
        shank_slice = self.sensor_slices.get("shank")
        if thigh_slice is None or shank_slice is None:
            raise ValueError("sensor_slices must contain 'thigh' and 'shank' entries")

        thigh = x_in[..., thigh_slice]  # (B, T, C_thigh)
        shank = x_in[..., shank_slice]  # (B, T, C_shank)

        # Expect schema: [ax, ay, az, gx, gy, gz] per sensor
        if thigh.shape[-1] < 6 or shank.shape[-1] < 6:
            raise ValueError("Each sensor slice must have at least 6 channels (3 accel + 3 gyro)")

        # Accel and gyro blocks
        a_thigh = thigh[..., 0:3]
        g_thigh = thigh[..., 3:6]
        a_shank = shank[..., 0:3]
        g_shank = shank[..., 3:6]

        # Magnitudes per sensor
        a_mag_thigh = torch.linalg.vector_norm(a_thigh, dim=-1, keepdim=True)  # (B, T, 1)
        a_mag_shank = torch.linalg.vector_norm(a_shank, dim=-1, keepdim=True)  # (B, T, 1)
        g_mag_thigh = torch.linalg.vector_norm(g_thigh, dim=-1, keepdim=True)  # (B, T, 1)
        g_mag_shank = torch.linalg.vector_norm(g_shank, dim=-1, keepdim=True)  # (B, T, 1)

        # Inter-sensor difference magnitudes
        a_mag_diff = torch.linalg.vector_norm(a_thigh - a_shank, dim=-1, keepdim=True)  # (B, T, 1)
        g_mag_diff = torch.linalg.vector_norm(g_thigh - g_shank, dim=-1, keepdim=True)  # (B, T, 1)

        # Temporal first differences of accel magnitude per sensor
        # diff(t) = value(t) - value(t-1), diff(0) = 0
        zero = torch.zeros_like(a_mag_thigh[:, 0:1, :])
        a_mag_thigh_dt = torch.cat([zero, a_mag_thigh[:, 1:, :] - a_mag_thigh[:, :-1, :]], dim=1)
        zero_shank = torch.zeros_like(a_mag_shank[:, 0:1, :])
        a_mag_shank_dt = torch.cat([zero_shank, a_mag_shank[:, 1:, :] - a_mag_shank[:, :-1, :]], dim=1)

        extras = [
            a_mag_thigh,
            a_mag_shank,
            g_mag_thigh,
            g_mag_shank,
            a_mag_diff,
            g_mag_diff,
            a_mag_thigh_dt,
            a_mag_shank_dt,
        ]

        x_out = torch.cat([x_in] + extras, dim=-1)

        if squeeze_batch:
            return x_out.squeeze(0)
        return x_out
