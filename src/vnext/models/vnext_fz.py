from __future__ import annotations

from typing import Dict

import torch
from torch import nn


def _sinusoidal_positional_encoding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    if dim <= 0:
        raise ValueError("dim must be positive")
    pe = torch.zeros(length, dim, dtype=torch.float32, device=device)
    pos = torch.arange(0, length, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=device) * (-torch.log(torch.tensor(10000.0, device=device)) / dim)
    )
    pe[:, 0::2] = torch.sin(pos * div_term)
    if dim % 2 == 1:
        pe[:, 1::2] = torch.cos(pos * div_term[:-1])
    else:
        pe[:, 1::2] = torch.cos(pos * div_term)
    return pe


class _ResidualTCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        padding = (kernel_size // 2) * dilation
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding, dilation=dilation)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        y = self.conv1(x)
        y = self.act(y)
        y = self.dropout(y)
        y = self.conv2(y)
        y = self.dropout(y)
        y = y + x
        y = y.transpose(1, 2)
        y = self.ln(y)
        y = y.transpose(1, 2)
        y = self.act(y)
        return y


class PerSensorEncoder(nn.Module):
    """Simple temporal encoder for a single sensor.

    Uses Conv1d over time to capture short-range temporal structure.
    Input:  (B, T, C)
    Output: (B, T, H)
    """

    def __init__(self, in_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # x: (B, T, C) → Conv1d expects (B, C, T)
        x = x.transpose(1, 2)
        y = self.net(x)
        return y.transpose(1, 2)  # (B, T, H)


class VNextFzModel(nn.Module):
    """Temporal Fz-only vNext model using dual IMUs.

    - Separately encodes thigh and shank signals.
    - Fuses per-sensor features per time step.
    - Outputs a vertical GRF trajectory y_hat of shape (B, T, 1).
    """

    def __init__(
        self,
        in_channels: int,
        sensor_slices: Dict[str, slice],
        per_sensor_hidden: int = 32,
        fusion_hidden: int = 64,
        backbone: str = "baseline_mlp",
        dropout: float = 0.0,
        tcn_blocks: int = 5,
        transformer_layers: int = 3,
        transformer_d_model: int = 96,
        transformer_heads: int = 4,
    ) -> None:
        super().__init__()
        self.sensor_slices = sensor_slices

        thigh_slice = sensor_slices.get("thigh")
        shank_slice = sensor_slices.get("shank")
        if thigh_slice is None or shank_slice is None:
            raise ValueError("sensor_slices must contain 'thigh' and 'shank' entries")

        thigh_in = thigh_slice.stop - thigh_slice.start
        shank_in = shank_slice.stop - shank_slice.start

        self.backbone = str(backbone).lower()
        if self.backbone not in {"baseline_mlp", "tcn", "transformer_small"}:
            raise ValueError(f"Unsupported backbone '{backbone}'")

        self.dropout_p = float(dropout)

        self.thigh_encoder = None
        self.shank_encoder = None
        self.fusion = None
        self.in_proj = None
        self.tcn = None
        self.tcn_out = None
        self.tf_in = None
        self.tf = None
        self.tf_out = None

        if self.backbone == "baseline_mlp":
            self.thigh_encoder = PerSensorEncoder(thigh_in, per_sensor_hidden)
            self.shank_encoder = PerSensorEncoder(shank_in, per_sensor_hidden)
            fusion_in = per_sensor_hidden * 2
            self.fusion = nn.Sequential(
                nn.Linear(fusion_in, fusion_hidden),
                nn.ReLU(),
                nn.Dropout(self.dropout_p),
                nn.Linear(fusion_hidden, 1),
            )
        elif self.backbone == "tcn":
            self.in_proj = nn.Conv1d(in_channels, fusion_hidden, kernel_size=1)
            blocks = []
            n_blocks = int(tcn_blocks)
            if n_blocks < 1:
                raise ValueError("tcn_blocks must be >= 1")
            for i in range(n_blocks):
                blocks.append(
                    _ResidualTCNBlock(
                        channels=fusion_hidden,
                        kernel_size=3,
                        dilation=2**i,
                        dropout=self.dropout_p,
                    )
                )
            self.tcn = nn.Sequential(*blocks)
            self.tcn_out = nn.Conv1d(fusion_hidden, 1, kernel_size=1)
        else:
            d_model = int(transformer_d_model)
            n_heads = int(transformer_heads)
            n_layers = int(transformer_layers)
            if d_model <= 0 or n_heads <= 0 or n_layers <= 0:
                raise ValueError("transformer_d_model/heads/layers must be positive")
            self.tf_in = nn.Linear(in_channels, d_model)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=max(4 * d_model, d_model),
                dropout=self.dropout_p,
                batch_first=True,
                norm_first=True,
            )
            self.tf = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.tf_out = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.backbone == "baseline_mlp":
            if self.thigh_encoder is None or self.shank_encoder is None or self.fusion is None:
                raise RuntimeError("baseline_mlp backbone not initialized")
            thigh_x = x[:, :, self.sensor_slices["thigh"]]  # (B, T, C_thigh)
            shank_x = x[:, :, self.sensor_slices["shank"]]  # (B, T, C_shank)
            thigh_h = self.thigh_encoder(thigh_x)  # (B, T, H)
            shank_h = self.shank_encoder(shank_x)  # (B, T, H)
            h = torch.cat([thigh_h, shank_h], dim=-1)  # (B, T, 2H)
            B, T, F = h.shape
            h_flat = h.reshape(B * T, F)
            y_flat = self.fusion(h_flat)
            y = y_flat.reshape(B, T, 1)
            return y

        if self.backbone == "tcn":
            if self.in_proj is None or self.tcn is None or self.tcn_out is None:
                raise RuntimeError("tcn backbone not initialized")
            y = x.transpose(1, 2)  # (B, C, T)
            y = self.in_proj(y)
            y = self.tcn(y)
            y = self.tcn_out(y)
            return y.transpose(1, 2)  # (B, T, 1)

        if self.tf_in is None or self.tf is None or self.tf_out is None:
            raise RuntimeError("transformer_small backbone not initialized")
        h = self.tf_in(x)  # (B, T, D)
        pe = _sinusoidal_positional_encoding(h.shape[1], h.shape[2], device=h.device)
        h = h + pe.unsqueeze(0)
        h = self.tf(h)
        y = self.tf_out(h)
        return y
