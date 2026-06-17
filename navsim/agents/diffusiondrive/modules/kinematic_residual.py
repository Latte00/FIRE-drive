import torch
import torch.nn as nn


class KinematicResidualHead(nn.Module):
    """Small MLP to predict residual correction for next-step kinematics."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def constant_accel_next_xy(status_feature: torch.Tensor, dt: float) -> torch.Tensor:
    """Predict next-step delta (x, y) using constant acceleration in ego frame."""
    if status_feature.dim() == 1:
        status_feature = status_feature.unsqueeze(0)
    device = status_feature.device
    dtype = status_feature.dtype
    dt = float(dt) if dt > 0 else 1.0
    velocity = (
        status_feature[:, 4:6]
        if status_feature.shape[-1] >= 6
        else torch.zeros((status_feature.shape[0], 2), device=device, dtype=dtype)
    )
    acceleration = (
        status_feature[:, 6:8]
        if status_feature.shape[-1] >= 8
        else torch.zeros((status_feature.shape[0], 2), device=device, dtype=dtype)
    )
    return velocity * dt + 0.5 * acceleration * dt * dt
