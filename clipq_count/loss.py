"""Smooth-L1 on count + Smooth-L1 on log-count + MSE on density map."""
import torch.nn as nn


class CombinedCountingLoss(nn.Module):
    def __init__(self, log_weight: float = 0.5, density_weight: float = 0.1):
        super().__init__()
        self.log_weight = log_weight
        self.density_weight = density_weight
        self.smooth_l1 = nn.SmoothL1Loss()
        self.mse = nn.MSELoss()

    def forward(self, count_pred, count, log_pred=None, log_target=None, density_pred=None, density_target=None):
        loss = self.smooth_l1(count_pred, count)
        if log_pred is not None and log_target is not None:
            loss = loss + self.log_weight * self.smooth_l1(log_pred, log_target)
        if density_pred is not None and density_target is not None:
            loss = loss + self.density_weight * self.mse(density_pred, density_target)
        return loss
