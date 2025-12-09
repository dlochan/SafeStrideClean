from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class ChannelNormStats:
    """Per-channel normalization statistics.

    mean and std are 1D tensors of shape (C,) for C channels.
    """

    mean: torch.Tensor
    std: torch.Tensor

    def to_dict(self) -> Dict[str, list[float]]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, list[float]]) -> "ChannelNormStats":
        mean = torch.tensor(d["mean"], dtype=torch.float32)
        std = torch.tensor(d["std"], dtype=torch.float32)
        return cls(mean=mean, std=std)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor x along its last dimension.

        Parameters
        ----------
        x : torch.Tensor
            Tensor with shape (..., C).
        """
        return (x - self.mean) / (self.std.clamp(min=1e-6))
