"""LeadLagNet: siamese 1D-CNN encoder + correlation fusion + two heads.

Input  x: (N, 2, L)  — two z-scored windows [A, B]
Output lag_logits: (N, 21) over tau in [-MAX_LAG, +MAX_LAG]  (tau > 0: A leads B)
       rho_hat:    (N,)   in [-1, 1]

The scale factor beta is NOT a learned head: with z-scored inputs the OLS
slope equals rho, so beta in original units is recovered analytically as
beta = rho * std(B) / std(A) at readout time.
"""

import torch
import torch.nn as nn


class LeadLagNet(nn.Module):
    def __init__(self, max_lag: int = 10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(24, 48, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        # fusion sees [F_A, F_B, F_A * F_B]; the product term carries
        # correlation-like evidence at every time step
        self.fusion = nn.Sequential(
            nn.Conv1d(144, 96, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.Conv1d(96, 96, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(192, 128),
            nn.GELU(),
        )
        self.lag_head = nn.Linear(128, 2 * max_lag + 1)
        self.rho_head = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        fa = self.encoder(x[:, 0:1])
        fb = self.encoder(x[:, 1:2])
        h = self.fusion(torch.cat([fa, fb, fa * fb], dim=1))
        h = torch.cat([h.mean(dim=-1), h.amax(dim=-1)], dim=1)
        h = self.trunk(h)
        return self.lag_head(h), torch.tanh(self.rho_head(h)).squeeze(-1)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
