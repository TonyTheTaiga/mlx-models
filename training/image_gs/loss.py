from __future__ import annotations

import mlx.core as mx


def gaussian_kernel_1d(size: int, sigma: float | None = None) -> mx.array:
    if sigma is None:
        sigma = 1.5 if size == 11 else 0.15 * float(size)
    half = (size - 1) * 0.5
    coords = mx.arange(size, dtype=mx.float32) - mx.array(half, dtype=mx.float32)
    kernel = mx.exp(-(coords * coords) / (2.0 * sigma * sigma))
    kernel = kernel / mx.sum(kernel)
    return kernel


def gaussian_blur_2d(xb: mx.array, size: int = 11, sigma: float | None = None) -> mx.array:
    k = gaussian_kernel_1d(size, sigma)
    pad = size // 2
    _, h, w, _ = xb.shape
    xh = mx.pad(xb, [(0, 0), (0, 0), (pad, pad), (0, 0)], mode="edge")
    out_h = mx.zeros_like(xb)
    for i in range(size):
        out_h = out_h + k[i] * xh[:, :, i : i + w, :]
    xv = mx.pad(out_h, [(0, 0), (pad, pad), (0, 0), (0, 0)], mode="edge")
    out_v = mx.zeros_like(out_h)
    for i in range(size):
        out_v = out_v + k[i] * xv[:, i : i + h, :, :]
    return out_v


def ssim_map(x: mx.array, y: mx.array, max_val: float = 1.0, window_size: int = 11) -> mx.array:
    x32 = x.astype(mx.float32)
    y32 = y.astype(mx.float32)
    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * max_val) * (k1 * max_val)
    c2 = (k2 * max_val) * (k2 * max_val)
    xb = x32[None, ...]
    yb = y32[None, ...]
    mu_x = gaussian_blur_2d(xb, size=window_size)[0]
    mu_y = gaussian_blur_2d(yb, size=window_size)[0]
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = gaussian_blur_2d(xb * xb, size=window_size)[0] - mu_x2
    sigma_y2 = gaussian_blur_2d(yb * yb, size=window_size)[0] - mu_y2
    sigma_xy = gaussian_blur_2d(xb * yb, size=window_size)[0] - mu_xy
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
