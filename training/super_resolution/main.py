#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterator, Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import cv2

from networks.autoencoders.super_resolution.model import SuperResolution
from networks.utils.perceputal_loss import PerceptualLoss
from networks.vgg16.model import VGG16

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VGG_WEIGHTS = ROOT / "networks" / "vgg16" / "weights.npz"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def list_image_files(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"dataset path {directory} was not found")
    files = [p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not files:
        raise RuntimeError(f"no supported images were found under {directory}")
    return sorted(files)


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_image(image: np.ndarray, width: int, height: int, mode: str = "bicubic") -> np.ndarray:
    if mode == "area":
        interpolation = cv2.INTER_AREA
    elif mode == "linear":
        interpolation = cv2.INTER_LINEAR
    else:
        interpolation = cv2.INTER_CUBIC
    return cv2.resize(image, (width, height), interpolation=interpolation)


class SuperResolutionDataset:
    def __init__(
        self,
        directory: Path,
        hr_patch: int,
        upscale: int,
        augment: bool = True,
        seed: int | None = None,
    ):
        if hr_patch % upscale != 0:
            raise ValueError("hr_patch must be divisible by the upscale factor")
        self.hr_patch = hr_patch
        self.lr_patch = hr_patch // upscale
        self.augment = augment
        self.paths = list_image_files(directory)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.paths)

    def _random_crop(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        if h < self.hr_patch or w < self.hr_patch:
            min_h = max(1, h)
            min_w = max(1, w)
            scale = max(self.hr_patch / min_h, self.hr_patch / min_w)
            nh = max(self.hr_patch, int(math.ceil(h * scale)))
            nw = max(self.hr_patch, int(math.ceil(w * scale)))
            image = resize_image(image, nw, nh, mode="bicubic")
            h, w = image.shape[:2]
        y = 0 if h == self.hr_patch else int(self.rng.integers(0, h - self.hr_patch + 1))
        x = 0 if w == self.hr_patch else int(self.rng.integers(0, w - self.hr_patch + 1))
        patch = image[y : y + self.hr_patch, x : x + self.hr_patch]
        if self.augment and self.rng.random() < 0.5:
            patch = patch[:, ::-1]
        if self.augment and self.rng.random() < 0.5:
            patch = patch[::-1]
        return patch

    def _create_pair(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        hr_patch = self._random_crop(image)
        lr_patch = resize_image(hr_patch, self.lr_patch, self.lr_patch, mode="area")
        hr_patch = hr_patch.astype(np.float32) / 255.0
        lr_patch = lr_patch.astype(np.float32) / 255.0
        return lr_patch, hr_patch

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        image = read_image(self.paths[idx])
        return self._create_pair(image)

    def sample_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        indices = self.rng.integers(0, len(self.paths), size=batch_size)
        lr_batch = []
        hr_batch = []
        for index in indices:
            lr_patch, hr_patch = self[int(index)]
            lr_batch.append(lr_patch)
            hr_batch.append(hr_patch)
        return np.asarray(lr_batch, dtype=np.float32), np.asarray(hr_batch, dtype=np.float32)


def iterate_minibatches(
    dataset: SuperResolutionDataset, batch_size: int, steps: int | None
) -> Iterator[tuple[mx.array, mx.array]]:
    if steps is not None:
        for _ in range(steps):
            lr_np, hr_np = dataset.sample_batch(batch_size)
            yield mx.array(lr_np, dtype=mx.float32), mx.array(hr_np, dtype=mx.float32)
        return

    order = np.random.permutation(len(dataset))
    for start in range(0, len(dataset), batch_size):
        batch_idx = order[start : start + batch_size]
        lr_batch = []
        hr_batch = []
        for idx in batch_idx:
            lr_np, hr_np = dataset[int(idx)]
            lr_batch.append(lr_np)
            hr_batch.append(hr_np)
        if not lr_batch:
            continue
        yield mx.array(np.asarray(lr_batch, dtype=np.float32)), mx.array(
            np.asarray(hr_batch, dtype=np.float32)
        )


def gaussian_kernel(size: int, sigma: float | None = None) -> mx.array:
    if sigma is None:
        sigma = 1.5 if size == 11 else 0.15 * float(size)
    coords = mx.arange(size, dtype=mx.float32) - mx.array((size - 1) * 0.5, dtype=mx.float32)
    kernel = mx.exp(-(coords * coords) / (2.0 * sigma * sigma))
    return kernel / mx.sum(kernel)


def gaussian_blur_2d(x: mx.array, size: int = 11, sigma: float | None = None) -> mx.array:
    squeeze = False
    if x.ndim == 3:
        squeeze = True
        x = x[None, ...]
    kernel = gaussian_kernel(size, sigma)
    pad = size // 2
    _, height, width, _ = x.shape
    xh = mx.pad(x, [(0, 0), (0, 0), (pad, pad), (0, 0)], mode="edge")
    horiz = mx.zeros_like(x)
    for i in range(size):
        horiz = horiz + kernel[i] * xh[:, :, i : i + width, :]
    xv = mx.pad(horiz, [(0, 0), (pad, pad), (0, 0), (0, 0)], mode="edge")
    blur = mx.zeros_like(horiz)
    for i in range(size):
        blur = blur + kernel[i] * xv[:, i : i + height, :, :]
    if squeeze:
        return blur[0]
    return blur


def ssim_map(x: mx.array, y: mx.array, max_val: float = 1.0, window_size: int = 11) -> mx.array:
    squeeze = False
    if x.ndim == 3:
        squeeze = True
        x = x[None, ...]
        y = y[None, ...]

    x32 = x.astype(mx.float32)
    y32 = y.astype(mx.float32)
    k1 = 0.01
    k2 = 0.03
    c1 = (k1 * max_val) * (k1 * max_val)
    c2 = (k2 * max_val) * (k2 * max_val)

    mu_x = gaussian_blur_2d(x32, size=window_size)
    mu_y = gaussian_blur_2d(y32, size=window_size)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = gaussian_blur_2d(x32 * x32, size=window_size) - mu_x2
    sigma_y2 = gaussian_blur_2d(y32 * y32, size=window_size) - mu_y2
    sigma_xy = gaussian_blur_2d(x32 * y32, size=window_size) - mu_xy

    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim = numerator / denominator
    ssim = mx.mean(ssim, axis=-1, keepdims=True)
    if squeeze:
        return ssim[0]
    return ssim


def structural_loss(
    prediction: mx.array,
    target: mx.array,
    l1_weight: float,
    ssim_weight: float,
) -> tuple[mx.array, mx.array, mx.array]:
    pred32 = prediction.astype(mx.float32)
    target32 = target.astype(mx.float32)
    l1 = mx.mean(mx.abs(pred32 - target32))
    ssim = mx.mean(ssim_map(pred32, target32))
    loss = l1_weight * l1 + ssim_weight * (mx.array(1.0, dtype=mx.float32) - ssim)
    return loss, ssim, l1


def compute_psnr(prediction: mx.array, target: mx.array, max_val: float = 1.0) -> float:
    diff = prediction.astype(mx.float32) - target.astype(mx.float32)
    mse = mx.mean(diff * diff).item()
    if mse <= 1e-10 or not math.isfinite(mse):
        return 99.0
    return float(20.0 * math.log10(max_val) - 10.0 * math.log10(mse))


def make_loss_fn(
    perceptual_loss: PerceptualLoss,
    layers: Sequence[int],
    weights: Sequence[float] | None,
    l1_weight: float,
    ssim_weight: float,
    structural_weight: float,
    perceptual_weight: float,
):
    def loss_fn(model: SuperResolution, lr_batch: mx.array, hr_batch: mx.array):
        sr = mx.clip(model(lr_batch), 0.0, 1.0)
        struct_loss, ssim_value, l1_value = structural_loss(sr, hr_batch, l1_weight, ssim_weight)
        perc_loss = perceptual_loss(
            sr,
            hr_batch,
            layers=list(layers),
            weights=list(weights) if weights else None,
        )
        total = structural_weight * struct_loss + perceptual_weight * perc_loss
        return total, (sr, struct_loss, perc_loss, ssim_value, l1_value)

    return loss_fn


def train(args: argparse.Namespace) -> None:
    args.dataset = args.dataset.expanduser()
    args.vgg_weights = args.vgg_weights.expanduser()
    if args.checkpoint_dir is not None:
        args.checkpoint_dir = args.checkpoint_dir.expanduser()
    mx.random.seed(args.seed)
    dataset = SuperResolutionDataset(
        directory=args.dataset,
        hr_patch=args.hr_patch_size,
        upscale=args.upscale,
        augment=not args.no_augment,
        seed=args.seed,
    )

    # Architecture depth/width are fixed to stable defaults for now.
    model = SuperResolution(upscale=args.upscale, num_encoder_layers=5, latent_dim=256)
    mx.eval(model.parameters())

    feature_extractor = VGG16(num_classes=1000)
    feature_extractor.load_weights(str(args.vgg_weights), strict=True)
    perceptual_loss = PerceptualLoss(feature_extractor)

    perceptual_layers = list(args.perceptual_layers)
    per_layer_weights: list[float] | None = None
    if args.perceptual_weights:
        if len(args.perceptual_weights) != len(perceptual_layers):
            raise ValueError("number of perceptual weights must match number of perceptual layers")
        per_layer_weights = list(args.perceptual_weights)

    loss_fn = make_loss_fn(
        perceptual_loss=perceptual_loss,
        layers=perceptual_layers,
        weights=per_layer_weights,
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        structural_weight=args.structural_weight,
        perceptual_weight=args.perceptual_weight,
    )
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=args.learning_rate)

    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch is not None and steps_per_epoch <= 0:
        steps_per_epoch = None
    if steps_per_epoch is None:
        steps_per_epoch = max(1, len(dataset) // args.batch_size)

    best_psnr = -float("inf")
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        running_struct = 0.0
        running_perc = 0.0
        running_ssim = 0.0
        running_l1 = 0.0
        running_psnr = 0.0
        num_steps = 0

        for step, (lr_batch, hr_batch) in enumerate(
            iterate_minibatches(dataset, args.batch_size, steps_per_epoch),
            start=1,
        ):
            (loss_value, (sr, struct_loss, perc_loss, ssim_value, l1_value)), grads = loss_and_grad_fn(
                model, lr_batch, hr_batch
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            loss_item = loss_value.item()
            struct_item = struct_loss.item()
            perc_item = perc_loss.item()
            ssim_item = ssim_value.item()
            l1_item = l1_value.item()
            psnr_item = compute_psnr(sr, hr_batch)

            running_loss += loss_item
            running_struct += struct_item
            running_perc += perc_item
            running_ssim += ssim_item
            running_l1 += l1_item
            running_psnr += psnr_item
            num_steps += 1

            if step % args.log_every == 0:
                print(
                    f"epoch {epoch:03d} step {step:04d}/{steps_per_epoch:04d} "
                    f"loss={loss_item:.4f} struct={struct_item:.4f} perc={perc_item:.4f} "
                    f"ssim={ssim_item:.4f} psnr={psnr_item:.2f} l1={l1_item:.4f}"
                )

        epoch_loss = running_loss / max(1, num_steps)
        epoch_struct = running_struct / max(1, num_steps)
        epoch_perc = running_perc / max(1, num_steps)
        epoch_ssim = running_ssim / max(1, num_steps)
        epoch_l1 = running_l1 / max(1, num_steps)
        epoch_psnr = running_psnr / max(1, num_steps)

        print(
            f"[epoch {epoch:03d}/{args.epochs:03d}] "
            f"loss={epoch_loss:.4f} struct={epoch_struct:.4f} perc={epoch_perc:.4f} "
            f"ssim={epoch_ssim:.4f} psnr={epoch_psnr:.2f} l1={epoch_l1:.4f}"
        )

        if checkpoint_dir is not None:
            ckpt_path = checkpoint_dir / f"superres_epoch_{epoch:04d}.npz"
            model.save_weights(str(ckpt_path))
            if epoch_psnr > best_psnr:
                best_psnr = epoch_psnr
                model.save_weights(str(checkpoint_dir / "best.npz"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the super-resolution autoencoder.")
    parser.add_argument("--dataset", type=Path, required=True, help="path to HR training images")
    parser.add_argument("--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-2, help="optimizer learning rate")
    parser.add_argument("--upscale", type=int, default=4, help="super-resolution scale factor")
    parser.add_argument(
        "--hr-patch-size",
        type=int,
        default=192,
        help="size of cropped HR patches (must be divisible by the upscale factor)",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=None,
        help="optional fixed number of steps per epoch (defaults to len(dataset)//batch_size)",
    )
    parser.add_argument("--no-augment", action="store_true", help="disable random flips during training")
    parser.add_argument("--seed", type=int, default=42, help="PRNG seed")
    parser.add_argument("--log-every", type=int, default=10, help="steps between logging updates")
    parser.add_argument(
        "--perceptual-layers",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="indices of VGG feature maps to use inside the perceptual loss",
    )
    parser.add_argument(
        "--perceptual-weights",
        type=float,
        nargs="+",
        default=None,
        help="optional weights matching the perceptual layers",
    )
    parser.add_argument("--l1-weight", type=float, default=0.15, help="weight applied to pixel L1 loss")
    parser.add_argument("--ssim-weight", type=float, default=0.85, help="weight applied to (1 - SSIM)")
    parser.add_argument("--structural-weight", type=float, default=1.0, help="global weight for structural loss")
    parser.add_argument("--perceptual-weight", type=float, default=0.1, help="global weight for perceptual loss")
    parser.add_argument(
        "--vgg-weights",
        type=Path,
        default=DEFAULT_VGG_WEIGHTS,
        help="path to pre-trained VGG16 weights used by the perceptual loss",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="optional directory where model checkpoints will be stored",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
