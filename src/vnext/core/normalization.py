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

    def to(self, device: torch.device | str) -> "ChannelNormStats":
        """Return a copy of these stats on the requested device.

        Normalization stats are typically stored/loaded on CPU. This helper
        allows moving them to the same device as the model or inputs when
        desired, without mutating the original instance.
        """

        dev = torch.device(device)
        return ChannelNormStats(mean=self.mean.to(dev), std=self.std.to(dev))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a tensor x along its last dimension.

        Parameters
        ----------
        x : torch.Tensor
            Tensor with shape (..., C).
        """
        # Ensure normalization stats live on the same device as the input.
        # mean/std are often created/loaded on CPU, while x may be on CUDA.
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std.clamp(min=1e-6)


@dataclass
class TargetNormStats:
    kind: str
    center: torch.Tensor
    scale: torch.Tensor

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "TargetNormStats":
        kind = str(d.get("kind", "none"))
        center = torch.tensor(d.get("center", [0.0]), dtype=torch.float32)
        scale = torch.tensor(d.get("scale", [1.0]), dtype=torch.float32)
        return cls(kind=kind, center=center, scale=scale)

    def normalize(self, y: torch.Tensor) -> torch.Tensor:
        c = self.center.to(y.device)
        s = self.scale.to(y.device)
        return (y - c) / s.clamp(min=1e-6)

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:
        c = self.center.to(y.device)
        s = self.scale.to(y.device)
        return (y * s) + c
