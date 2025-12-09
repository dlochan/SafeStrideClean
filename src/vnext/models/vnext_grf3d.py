from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .vnext_fz import PerSensorEncoder


class VNextGRF3DModel(nn.Module):
    """Temporal GRF-3D vNext model using dual IMUs.

    - Separately encodes thigh and shank signals.
    - Fuses per-sensor features per time step.
    - Outputs [Fx, Fy, Fz] per time step with shape (B, T, 3).
    """

    def __init__(
        self,
        in_channels: int,
        sensor_slices: Dict[str, slice],
        per_sensor_hidden: int = 32,
        fusion_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.sensor_slices = sensor_slices

        thigh_slice = sensor_slices.get("thigh")
        shank_slice = sensor_slices.get("shank")
        if thigh_slice is None or shank_slice is None:
            raise ValueError("sensor_slices must contain 'thigh' and 'shank' entries")

        thigh_in = thigh_slice.stop - thigh_slice.start
        shank_in = shank_slice.stop - shank_slice.start

        self.thigh_encoder = PerSensorEncoder(thigh_in, per_sensor_hidden)
        self.shank_encoder = PerSensorEncoder(shank_in, per_sensor_hidden)

        fusion_in = per_sensor_hidden * 2
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Linear(fusion_hidden, 3),  # Fx, Fy, Fz
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x: (B, T, C)
        thigh_x = x[:, :, self.sensor_slices["thigh"]]  # (B, T, C_thigh)
        shank_x = x[:, :, self.sensor_slices["shank"]]  # (B, T, C_shank)

        thigh_h = self.thigh_encoder(thigh_x)  # (B, T, H)
        shank_h = self.shank_encoder(shank_x)  # (B, T, H)

        h = torch.cat([thigh_h, shank_h], dim=-1)  # (B, T, 2H)
        B, T, F = h.shape
        h_flat = h.reshape(B * T, F)
        y_flat = self.fusion(h_flat)
        y = y_flat.reshape(B, T, 3)
        return y
