from __future__ import annotations

import math

import torch
from torch import Tensor
import torch.nn.functional as F


def _unit_range(image: Tensor) -> Tensor:
    return image.float().add(1.0).div(2.0).clamp(0.0, 1.0)


def psnr(prediction: Tensor, target: Tensor) -> Tensor:
    mse = (_unit_range(prediction) - _unit_range(target)).square().mean(dim=(1, 2, 3))
    return -10.0 * torch.log10(mse.clamp_min(1e-12))


def mae(prediction: Tensor, target: Tensor) -> Tensor:
    return (_unit_range(prediction) - _unit_range(target)).abs().mean(dim=(1, 2, 3))


def _gaussian_kernel(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> Tensor:
    coordinate = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    kernel = torch.exp(-(coordinate.square()) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return torch.outer(kernel, kernel)


def ssim(prediction: Tensor, target: Tensor, window_size: int = 11, sigma: float = 1.5) -> Tensor:
    x, y = _unit_range(prediction), _unit_range(target)
    channels = x.shape[1]
    kernel = _gaussian_kernel(window_size, sigma, x.device, x.dtype)
    kernel = kernel.expand(channels, 1, window_size, window_size)
    mu_x = F.conv2d(x, kernel, groups=channels)
    mu_y = F.conv2d(y, kernel, groups=channels)
    mu_x2, mu_y2, mu_xy = mu_x.square(), mu_y.square(), mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, kernel, groups=channels) - mu_x2
    sigma_y2 = F.conv2d(y * y, kernel, groups=channels) - mu_y2
    sigma_xy = F.conv2d(x * y, kernel, groups=channels) - mu_xy
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
        (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    )
    return score.flatten(1).mean(1)


class LPIPSMetric:
    def __init__(self, device: torch.device) -> None:
        try:
            import lpips
        except ImportError as error:
            raise RuntimeError("LPIPS requires `pip install lpips`") from error
        self.model = lpips.LPIPS(net="alex").to(device).eval()

    @torch.no_grad()
    def __call__(self, prediction: Tensor, target: Tensor) -> Tensor:
        return self.model(prediction, target, normalize=False).flatten()

