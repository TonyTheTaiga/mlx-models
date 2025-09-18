from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


def ssim_map(x: mx.array, y: mx.array, max_val: float = 1.0, window_size: int = 11) -> mx.array:
    x32 = x.astype(mx.float32)
    y32 = y.astype(mx.float32)

    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * max_val) * (k1 * max_val)
    c2 = (k2 * max_val) * (k2 * max_val)

    pad = window_size // 2
    pool = nn.AvgPool2d(kernel_size=window_size, stride=1, padding=pad)

    xb = x32[None, ...]
    yb = y32[None, ...]

    mu_x = pool(xb)[0]
    mu_y = pool(yb)[0]
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = pool(xb * xb)[0] - mu_x2
    sigma_y2 = pool(yb * yb)[0] - mu_y2
    sigma_xy = pool(xb * yb)[0] - mu_xy

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim = num / den
    ssim = mx.mean(ssim, axis=-1, keepdims=True)
    return ssim


def l1_ssim_loss(model, target_image: mx.array) -> tuple[mx.array, mx.array]:
    rendered = model(target_image)
    x = rendered.astype(mx.float32)
    y = target_image.astype(mx.float32)

    l1 = mx.mean(mx.abs(x - y), axis=-1, keepdims=True)
    l1_loss = mx.mean(l1)

    ssim = ssim_map(x, y, max_val=1.0)
    l_ssim = mx.mean(1.0 - ssim)

    loss_value = l1_loss + mx.array(0.1, dtype=mx.float32) * l_ssim
    return loss_value, rendered


__all__ = ["ssim_map", "l1_ssim_loss"]
