from __future__ import annotations

from typing import Tuple

import torch
from torch import nn


class VNextStubModel(nn.Module):
    """A minimal placeholder model for vNext.

    This model is *not* the real production architecture. It only serves
    to verify wiring from config → DataLoader → model forward pass.

    Inputs
    ------
    x : torch.FloatTensor
        Tensor of shape (batch, T, C) containing dual-IMU features.

    Outputs
    -------
    torch.FloatTensor
        Tensor of shape (batch, T, 1) representing a dummy vertical GRF
        prediction per time step.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        # Simple per-time-step linear projection C → 1
        self.proj = nn.Linear(in_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x: (B, T, C) → reshape for Linear
        b, t, c = x.shape
        x_flat = x.reshape(b * t, c)
        y_flat = self.proj(x_flat)
        y = y_flat.reshape(b, t, 1)
        return y
