from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


def _expand(value: Tensor, ndim: int = 4) -> Tensor:
    return value.reshape(value.shape[0], *([1] * (ndim - 1)))


@dataclass
class FlowBatch:
    xt: Tensor
    yt: Tensor
    t: Tensor
    target: Tensor
    weight: Tensor


class ResFlowProcess:
    """Equations (7)--(11) and four-step reverse Euler integration.

    The paper's displayed entropy-preserving schedule is followed literally:
    sigma_y(t) = beta / (1 - t + beta).  Consequently sigma_y(0) is
    beta/(1+beta), despite the prose also describing y_0 as zero.  See
    REPRODUCIBILITY.md for this paper inconsistency.
    """

    def __init__(self, beta: float = 10.0, gamma: float = 1.75) -> None:
        if beta <= 0 or gamma <= 0:
            raise ValueError("beta and gamma must be positive")
        self.beta = float(beta)
        self.gamma = float(gamma)

    def sigma_y(self, t: Tensor) -> Tensor:
        return self.beta / (1.0 - t + self.beta)

    def sigma_y_derivative(self, t: Tensor) -> Tensor:
        return self.beta / (1.0 - t + self.beta).square()

    def loss_weight(self, t: Tensor) -> Tensor:
        base = torch.cos((math.pi / 2.0) * (t - 2.0)) + 1.0
        return base.clamp_min(0.0).pow(self.gamma)

    def training_batch(
        self,
        x0: Tensor,
        x1: Tensor,
        *,
        t: Tensor | None = None,
        y1: Tensor | None = None,
    ) -> FlowBatch:
        if x0.shape != x1.shape:
            raise ValueError(f"HQ/LQ shapes differ: {x0.shape} vs {x1.shape}")
        batch = x0.shape[0]
        if t is None:
            t = torch.rand(batch, device=x0.device, dtype=x0.dtype)
        if y1 is None:
            y1 = torch.randn_like(x0)
        if t.shape != (batch,):
            raise ValueError(f"t must have shape ({batch},), got {t.shape}")

        tb = _expand(t, x0.ndim)
        xt = (1.0 - tb) * x0 + tb * x1
        yt = _expand(self.sigma_y(t), x0.ndim) * y1
        velocity_x = x1 - x0
        velocity_y = _expand(self.sigma_y_derivative(t), x0.ndim) * y1
        target = torch.cat((velocity_x, velocity_y), dim=1)
        return FlowBatch(xt, yt, t, target, self.loss_weight(t))

    def loss(self, prediction: Tensor, batch: FlowBatch) -> Tensor:
        if prediction.shape != batch.target.shape:
            raise ValueError(
                f"velocity shape {prediction.shape} != target {batch.target.shape}"
            )
        per_sample = (prediction - batch.target).square().flatten(1).mean(1)
        return (batch.weight * per_sample).mean()

    @torch.no_grad()
    def restore(
        self,
        model: nn.Module,
        x1: Tensor,
        *,
        steps: int = 4,
        y1: Tensor | None = None,
        clamp: bool = True,
    ) -> Tensor:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        x = x1.clone()
        y1 = torch.randn_like(x1) if y1 is None else y1
        times = torch.linspace(1.0, 0.0, steps + 1, device=x.device, dtype=x.dtype)
        for index in range(steps):
            t_value = times[index]
            next_t = times[index + 1]
            t = t_value.expand(x.shape[0])
            yt = _expand(self.sigma_y(t), x.ndim) * y1
            velocity = model(x, yt, t)
            velocity_x = velocity[:, : x.shape[1]]
            x = x + (next_t - t_value) * velocity_x
        return x.clamp(-1.0, 1.0) if clamp else x

