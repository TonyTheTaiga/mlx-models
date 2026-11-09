#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Sequence

import cv2
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml

from networks.autoencoders.super_resolution.discriminator import Discriminator, UNetDiscriminator
from networks.autoencoders.super_resolution.model import SuperResolution
from networks.autoencoders.super_resolution.rrdb import RRDBNet
from networks.utils.perceputal_loss import PerceptualLoss
from networks.vgg16.model import VGG16

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_VGG_WEIGHTS = ROOT / "networks" / "vgg16" / "weights.npz"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
CONFIG_PATH = Path(__file__).with_name("config.yaml")


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
        cache_images: bool = False,
    ):
        if hr_patch % upscale != 0:
            raise ValueError("hr_patch must be divisible by the upscale factor")
        self.hr_patch = hr_patch
        self.lr_patch = hr_patch // upscale
        self.augment = augment
        self.paths = list_image_files(directory)
        self.rng = np.random.default_rng(seed)
        # decoding a 2K PNG per crop dominated step time
        self.cache: dict[int, np.ndarray] | None = {} if cache_images else None

    def __len__(self) -> int:
        return len(self.paths)

    def _random_crop(self, image: np.ndarray) -> np.ndarray:
        h, w, _ = image.shape
        if h < self.hr_patch or w < self.hr_patch:
            scale = max(self.hr_patch / max(1, h), self.hr_patch / max(1, w))
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

    def _load(self, idx: int) -> np.ndarray:
        if self.cache is None:
            return read_image(self.paths[idx])
        if idx not in self.cache:
            self.cache[idx] = read_image(self.paths[idx])
        return self.cache[idx]

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self._create_pair(self._load(idx))

    def sample_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        lr_batch = []
        hr_batch = []
        for index in self.rng.integers(0, len(self.paths), size=batch_size):
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
        yield (
            mx.array(np.asarray(lr_batch, dtype=np.float32)),
            mx.array(np.asarray(hr_batch, dtype=np.float32)),
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


def load_validation_crops(
    directory: Path, crop: int, upscale: int
) -> tuple[np.ndarray, np.ndarray]:
    lr_crops, hr_crops = [], []
    for path in list_image_files(directory):
        image = read_image(path)
        h, w = image.shape[:2]
        size = min(crop, h, w) // upscale * upscale
        y, x = (h - size) // 2, (w - size) // 2
        hr = image[y : y + size, x : x + size]
        lr = resize_image(hr, size // upscale, size // upscale, mode="area")
        hr_crops.append(hr.astype(np.float32) / 255.0)
        lr_crops.append(lr.astype(np.float32) / 255.0)
    return np.stack(lr_crops), np.stack(hr_crops)


def evaluate(
    model: nn.Module,
    perceptual_loss: PerceptualLoss,
    lr_crops: np.ndarray,
    hr_crops: np.ndarray,
    layers: Sequence[int],
    weights: Sequence[float] | None,
    batch_size: int,
) -> tuple[float, float]:
    """Returns (mean PSNR, mean VGG feature distance)."""
    model.train(False)
    psnrs, distances = [], []
    for start in range(0, len(lr_crops), batch_size):
        lr_batch = mx.array(lr_crops[start : start + batch_size])
        hr_batch = mx.array(hr_crops[start : start + batch_size])
        sr = mx.clip(model(lr_batch), 0.0, 1.0)
        for i in range(sr.shape[0]):
            psnrs.append(compute_psnr(sr[i], hr_batch[i]))
        distance = perceptual_loss(sr, hr_batch, layers=list(layers), weights=weights)
        distances.append(distance.item() * sr.shape[0])
    model.train(True)
    return float(np.mean(psnrs)), float(np.sum(distances) / len(lr_crops))


def build_generator(args: SimpleNamespace) -> nn.Module:
    if args.generator == "rrdb":
        return RRDBNet(upscale=args.upscale, num_blocks=args.rrdb_blocks)
    # Architecture depth/width are fixed to stable defaults for now.
    return SuperResolution(upscale=args.upscale, num_encoder_layers=5, latent_dim=256)


def build_discriminator(args: SimpleNamespace) -> nn.Module:
    if args.discriminator == "unet":
        return UNetDiscriminator()
    return Discriminator()


def lsgan_discriminator_loss(real_logits: mx.array, fake_logits: mx.array) -> mx.array:
    """LSGAN discriminator loss: (D(real) - 1)^2 + (D(fake) + 1)^2"""
    return mx.mean((real_logits - 1.0) ** 2) + mx.mean((fake_logits + 1.0) ** 2)


def lsgan_generator_loss(fake_logits: mx.array) -> mx.array:
    """LSGAN generator loss: (D(fake) - 1)^2"""
    return mx.mean((fake_logits - 1.0) ** 2)


def make_loss_fn(
    perceptual_loss: PerceptualLoss,
    discriminator: nn.Module | None,
    layers: Sequence[int],
    weights: Sequence[float] | None,
    l1_weight: float,
    ssim_weight: float,
    structural_weight: float,
    perceptual_weight: float,
    adversarial_weight: float,
):
    zero = mx.array(0.0, dtype=mx.float32)

    def loss_fn(model: nn.Module, lr_batch: mx.array, hr_batch: mx.array):
        # clipping kills gradients for out-of-range pixels, so only clip the VGG/D inputs
        sr = model(lr_batch)
        sr_clipped = mx.clip(sr, 0.0, 1.0)
        struct_loss, ssim_value, l1_value = structural_loss(sr, hr_batch, l1_weight, ssim_weight)
        perc_loss = zero
        if perceptual_weight > 0:
            perc_loss = perceptual_loss(
                sr_clipped,
                hr_batch,
                layers=list(layers),
                weights=list(weights) if weights else None,
            )
        adv_loss = zero
        if discriminator is not None and adversarial_weight > 0:
            adv_loss = lsgan_generator_loss(discriminator(sr_clipped))
        total = (
            structural_weight * struct_loss
            + perceptual_weight * perc_loss
            + adversarial_weight * adv_loss
        )
        return total, (sr_clipped, struct_loss, perc_loss, adv_loss, ssim_value, l1_value)

    return loss_fn


def train(args: SimpleNamespace) -> None:
    mx.random.seed(args.seed)
    # an unbounded buffer cache grows past the GPU working set in the GAN stage and pages
    mx.set_cache_limit(4 * 1024**3)
    dataset = SuperResolutionDataset(
        directory=args.dataset,
        hr_patch=args.hr_patch_size,
        upscale=args.upscale,
        augment=not args.no_augment,
        seed=args.seed,
        cache_images=args.cache_images,
    )

    model = build_generator(args)
    if args.init_checkpoint is not None:
        model.load_weights(str(args.init_checkpoint), strict=True)
        print(f"initialized generator from {args.init_checkpoint}")
    mx.eval(model.parameters())

    feature_extractor = VGG16(num_classes=1000)
    feature_extractor.load_weights(str(args.vgg_weights), strict=True)
    perceptual_loss = PerceptualLoss(feature_extractor)

    if args.perceptual_weights and len(args.perceptual_weights) != len(args.perceptual_layers):
        raise ValueError("number of perceptual weights must match number of perceptual layers")
    perceptual_weights = list(args.perceptual_weights) if args.perceptual_weights else None

    discriminator = None
    if args.adversarial_weight > 0:
        discriminator = build_discriminator(args)
        mx.eval(discriminator.parameters())

    loss_fn = make_loss_fn(
        perceptual_loss=perceptual_loss,
        discriminator=discriminator,
        layers=list(args.perceptual_layers),
        weights=perceptual_weights,
        l1_weight=args.l1_weight,
        ssim_weight=args.ssim_weight,
        structural_weight=args.structural_weight,
        perceptual_weight=args.perceptual_weight,
        adversarial_weight=args.adversarial_weight,
    )
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch is not None and steps_per_epoch <= 0:
        steps_per_epoch = None
    if steps_per_epoch is None:
        steps_per_epoch = max(1, len(dataset) // args.batch_size)
    total_steps = args.epochs * steps_per_epoch

    schedule = optim.cosine_decay(args.learning_rate, total_steps, args.learning_rate * 0.01)
    optimizer = optim.Adam(learning_rate=schedule, betas=[0.9, 0.99])

    if discriminator is not None:
        d_schedule = optim.cosine_decay(
            args.disc_learning_rate, total_steps, args.disc_learning_rate * 0.01
        )
        d_optimizer = optim.Adam(learning_rate=d_schedule, betas=[0.9, 0.99])
        d_loss_and_grad_fn = nn.value_and_grad(
            discriminator,
            lambda d, real, fake: lsgan_discriminator_loss(d(real), d(fake)),
        )

    val_lr, val_hr = None, None
    if args.val_dataset is not None:
        val_lr, val_hr = load_validation_crops(args.val_dataset, args.val_crop, args.upscale)
        print(f"validating on {len(val_lr)} center crops of {val_hr.shape[1]}px")

    sign = 1.0 if args.select_by == "psnr" else -1.0
    best_score = -float("inf")
    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        running = {"loss": 0.0, "struct": 0.0, "perc": 0.0, "adv": 0.0, "d": 0.0, "psnr": 0.0}
        num_steps = 0

        for step, (lr_batch, hr_batch) in enumerate(
            iterate_minibatches(dataset, args.batch_size, steps_per_epoch),
            start=1,
        ):
            (loss_value, (sr, struct_loss, perc_loss, adv_loss, ssim_value, l1_value)), grads = (
                loss_and_grad_fn(model, lr_batch, hr_batch)
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            d_item = 0.0
            if discriminator is not None:
                d_loss, d_grads = d_loss_and_grad_fn(discriminator, hr_batch, mx.stop_gradient(sr))
                d_optimizer.update(discriminator, d_grads)
                if isinstance(discriminator, UNetDiscriminator):
                    discriminator.power_iteration()
                mx.eval(discriminator.parameters(), d_optimizer.state)
                d_item = d_loss.item()

            values = {
                "loss": loss_value.item(),
                "struct": struct_loss.item(),
                "perc": perc_loss.item(),
                "adv": adv_loss.item(),
                "d": d_item,
                "psnr": compute_psnr(sr, hr_batch),
            }
            for key, value in values.items():
                running[key] += value
            num_steps += 1

            if step % args.log_every == 0:
                print(
                    f"epoch {epoch:03d} step {step:04d}/{steps_per_epoch:04d} "
                    f"loss={values['loss']:.4f} struct={values['struct']:.4f} "
                    f"perc={values['perc']:.4f} adv={values['adv']:.4f} d={values['d']:.4f} "
                    f"ssim={ssim_value.item():.4f} psnr={values['psnr']:.2f} "
                    f"l1={l1_value.item():.4f}"
                )

        means = {key: value / max(1, num_steps) for key, value in running.items()}
        summary = (
            f"[epoch {epoch:03d}/{args.epochs:03d}] "
            f"loss={means['loss']:.4f} struct={means['struct']:.4f} perc={means['perc']:.4f} "
            f"adv={means['adv']:.4f} d={means['d']:.4f} psnr={means['psnr']:.2f}"
        )

        score = means["psnr"] if args.select_by == "psnr" else -means["perc"]
        if val_lr is not None:
            val_psnr, val_distance = evaluate(
                model,
                perceptual_loss,
                val_lr,
                val_hr,
                args.perceptual_layers,
                perceptual_weights,
                args.batch_size,
            )
            summary += f" val_psnr={val_psnr:.2f} val_vgg={val_distance:.4f}"
            score = sign * (val_psnr if args.select_by == "psnr" else val_distance)
        print(summary)

        if checkpoint_dir is not None:
            ckpt_path = checkpoint_dir / f"superres_epoch_{epoch:04d}.npz"
            model.save_weights(str(ckpt_path))
            if discriminator is not None:
                discriminator.save_weights(str(checkpoint_dir / "discriminator.npz"))
            if score > best_score:
                best_score = score
                model.save_weights(str(checkpoint_dir / "best.npz"))


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def resolve_path(value: str | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def build_args_from_config(config_path: Path = CONFIG_PATH) -> SimpleNamespace:
    config = load_yaml_config(config_path)
    base = config_path.parent

    dataset_value = config.get("dataset")
    if dataset_value is None:
        raise ValueError("`dataset` must be set in the config file")
    dataset = resolve_path(dataset_value, base)

    vgg_weights = resolve_path(config.get("vgg_weights"), base) or DEFAULT_VGG_WEIGHTS
    checkpoint_dir = resolve_path(config.get("checkpoint_dir"), base)

    generator = config.get("generator", "resnet")
    if generator not in ("resnet", "rrdb"):
        raise ValueError("`generator` must be `resnet` or `rrdb`")
    discriminator = config.get("discriminator", "patch")
    if discriminator not in ("patch", "unet"):
        raise ValueError("`discriminator` must be `patch` or `unet`")

    select_by = config.get("select_by", "psnr")
    if select_by not in ("psnr", "perceptual"):
        raise ValueError("`select_by` must be `psnr` or `perceptual`")

    def as_int_list(value, default):
        seq = value if value is not None else default
        return [int(v) for v in seq]

    def as_float_list(value):
        if value is None:
            return None
        return [float(v) for v in value]

    steps_value = config.get("steps_per_epoch")
    learning_rate = float(config.get("learning_rate", 2e-4))
    return SimpleNamespace(
        dataset=dataset,
        val_dataset=resolve_path(config.get("val_dataset"), base),
        val_crop=int(config.get("val_crop", 256)),
        epochs=int(config.get("epochs", 50)),
        batch_size=int(config.get("batch_size", 16)),
        learning_rate=learning_rate,
        disc_learning_rate=float(config.get("disc_learning_rate", learning_rate)),
        upscale=int(config.get("upscale", 4)),
        generator=generator,
        rrdb_blocks=int(config.get("rrdb_blocks", 23)),
        discriminator=discriminator,
        hr_patch_size=int(config.get("hr_patch_size", 192)),
        steps_per_epoch=None if steps_value is None else int(steps_value),
        no_augment=bool(config.get("no_augment", False)),
        cache_images=bool(config.get("cache_images", False)),
        seed=int(config.get("seed", 42)),
        log_every=int(config.get("log_every", 10)),
        perceptual_layers=as_int_list(config.get("perceptual_layers"), [0, 1, 2, 3]),
        perceptual_weights=as_float_list(config.get("perceptual_weights")),
        l1_weight=float(config.get("l1_weight", 1.0)),
        ssim_weight=float(config.get("ssim_weight", 0.0)),
        structural_weight=float(config.get("structural_weight", 1.0)),
        perceptual_weight=float(config.get("perceptual_weight", 0.0)),
        adversarial_weight=float(config.get("adversarial_weight", 0.0)),
        select_by=select_by,
        init_checkpoint=resolve_path(config.get("init_checkpoint"), base),
        vgg_weights=vgg_weights,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    config_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else CONFIG_PATH
    train(build_args_from_config(config_path))
