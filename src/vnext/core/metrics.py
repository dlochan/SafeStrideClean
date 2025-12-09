from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class GRFMetrics:
    """Container for per-axis and aggregate GRF errors.

    The model may output 1 axis (Fz) or 3 axes (Fx, Fy, Fz).
    All fields are plain Python floats for easy serialization.
    """

    mse_per_axis: Dict[str, float]
    rmse_per_axis: Dict[str, float]
    mse_mean: float
    rmse_mean: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "mse_per_axis": self.mse_per_axis,
            "rmse_per_axis": self.rmse_per_axis,
            "mse_mean": self.mse_mean,
            "rmse_mean": self.rmse_mean,
        }


def compute_grf_metrics(y_hat: torch.Tensor, y_true: torch.Tensor, axes: str) -> GRFMetrics:
    """Compute per-axis and mean MSE/RMSE for GRF predictions.

    Parameters
    ----------
    y_hat : (B, T, D) predicted GRF
    y_true : (B, T, D) ground-truth GRF
    axes : "fz" or "3d"

    For "fz", treat it as a single axis "Fz".
    For "3d", assume channel order [Fx, Fy, Fz].
    """

    if y_hat.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: y_hat{tuple(y_hat.shape)} vs y_true{tuple(y_true.shape)}")

    B, T, D = y_hat.shape
    if axes == "fz" and D != 1:
        raise ValueError(f"Expected D=1 for axes='fz', got D={D}")
    if axes == "3d" and D != 3:
        raise ValueError(f"Expected D=3 for axes='3d', got D={D}")

    if axes == "fz":
        axis_names = ["Fz"]
    else:  # "3d"
        axis_names = ["Fx", "Fy", "Fz"]

    mse_per_axis: Dict[str, float] = {}
    rmse_per_axis: Dict[str, float] = {}

    # Flatten batch/time for per-axis metrics
    for i, name in enumerate(axis_names):
        y_hat_i = y_hat[..., i].reshape(-1)
        y_true_i = y_true[..., i].reshape(-1)
        mse = torch.mean((y_hat_i - y_true_i) ** 2).item()
        rmse = mse ** 0.5
        mse_per_axis[name] = float(mse)
        rmse_per_axis[name] = float(rmse)

    mse_mean = float(sum(mse_per_axis.values()) / len(mse_per_axis))
    rmse_mean = float(sum(rmse_per_axis.values()) / len(rmse_per_axis))

    return GRFMetrics(
        mse_per_axis=mse_per_axis,
        rmse_per_axis=rmse_per_axis,
        mse_mean=mse_mean,
        rmse_mean=rmse_mean,
    )
