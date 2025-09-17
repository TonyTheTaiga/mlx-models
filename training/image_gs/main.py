from __future__ import annotations

import cv2
import math
from pathlib import Path
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from networks.image_gs import build_model


def _gaussian_kernel1d(k: int = 11, sigma: float = 1.5) -> mx.array:
    r = k // 2
    xs = mx.arange(-r, r + 1, dtype=mx.float32)
    g = mx.exp(-0.5 * (xs / mx.array(sigma, dtype=mx.float32)) ** 2)
    g = g / mx.sum(g)
    return g


def _gaussian_blur2d(x: mx.array, k: int = 11, sigma: float = 1.5) -> mx.array:
    """Separable Gaussian blur with (k, sigma). x: (H,W,C) float32 -> same shape."""
    H, W, C = x.shape
    r = k // 2
    g = _gaussian_kernel1d(k, sigma)
    pad_h = mx.pad(x, [(0, 0), (r, r), (0, 0)], mode="edge")  # (H, W+2r, C)
    acc = mx.zeros((H, W, C), dtype=mx.float32)
    for i in range(k):
        acc = acc + g[i] * pad_h[:, i : i + W, :]
    pad_v = mx.pad(acc, [(r, r), (0, 0), (0, 0)], mode="edge")  # (H+2r, W, C)
    out = mx.zeros_like(acc)
    for j in range(k):
        out = out + g[j] * pad_v[j : j + H, :, :]
    return out


def ssim(img1: mx.array, img2: mx.array, k: int = 11, sigma: float = 1.5,
         c1: float = (0.01 ** 2), c2: float = (0.03 ** 2)) -> mx.array:
    w = mx.array([0.299, 0.587, 0.114], dtype=mx.float32)
    x = mx.sum(img1 * w[None, None, :], axis=-1, keepdims=True)
    y = mx.sum(img2 * w[None, None, :], axis=-1, keepdims=True)

    mu_x = _gaussian_blur2d(x, k, sigma)
    mu_y = _gaussian_blur2d(y, k, sigma)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = _gaussian_blur2d(x * x, k, sigma) - mu_x2
    sigma_y2 = _gaussian_blur2d(y * y, k, sigma) - mu_y2
    sigma_xy = _gaussian_blur2d(x * y, k, sigma) - mu_xy

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2) + 1e-6
    ssim_map = num / den
    return mx.mean(ssim_map)


def loss_fn(model, image_array, alpha_l1: float = 0.5):
    rgb = model(image_array)
    rgb32 = rgb.astype(mx.float32)
    img32 = image_array.astype(mx.float32)

    l1 = mx.mean(mx.abs(rgb32 - img32))
    ssim_val = ssim(rgb32, img32)
    loss = alpha_l1 * l1 + (1.0 - alpha_l1) * (1.0 - ssim_val)
    return loss


def psnr(x: mx.array, y: mx.array, max_val: float = 1.0) -> float:
    mse = mx.mean((x.astype(mx.float32) - y.astype(mx.float32)) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(max_val) - 10.0 * math.log10(mse))


def main(image_path: str, epochs: int = 2, lr: float = 1e-2, num_init: int = 2000, tile: int = 16, alpha_l1: float = 0.5, save: Path | None = None):
    image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR_RGB)
    if image is None:
        raise RuntimeError("failed to load image")

    image_array = mx.array(image).astype(mx.float32) / 255.0
    model = build_model(image_array, num_initial_gaussians=num_init, tile_size=tile)
    mx.eval(model.parameters())

    model(image_array)

    loss_and_grad_fn = nn.value_and_grad(model, lambda m, x: loss_fn(m, x, alpha_l1))
    optimizer = optim.Adam(learning_rate=lr)

    for epoch in range(epochs):
        loss, grads = loss_and_grad_fn(model, image_array)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        pred = model(image_array)
        cur_psnr = psnr(pred, image_array)
        print(f"epoch {epoch+1}/{epochs}  loss={loss.item():.6f}  psnr={cur_psnr:.2f} dB")

    if save is not None:
        out = (mx.clip(pred, 0.0, 1.0) * 255.0).astype(mx.uint8)
        out_np = np.array(out)
        # RGB -> BGR for OpenCV saving
        cv2.imwrite(str(save), out_np[:, :, ::-1])
        print(f"Saved reconstruction to {save}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num_init", type=int, default=2000)
    parser.add_argument("--tile", type=int, default=16)
    parser.add_argument("--alpha_l1", type=float, default=0.5)
    parser.add_argument("--save", type=Path, default=None)

    args = parser.parse_args()
    if not args.image.exists():
        raise FileNotFoundError("image not found!")

    main(
        image_path=args.image.as_posix(),
        epochs=args.epochs,
        lr=args.lr,
        num_init=args.num_init,
        tile=args.tile,
        alpha_l1=args.alpha_l1,
        save=args.save,
    )
