#!/usr/bin/env python3
"""
Upscale images with a trained super-resolution checkpoint.

The image is processed in overlapping tiles so full-size photos fit in memory; each tile keeps
only its interior, so the overlap hides seams. The architecture (RRDB or the original residual
network) is detected from the checkpoint's parameter names.

Usage:
    python training/super_resolution/upscale.py photo.jpg \
        --checkpoint training/super_resolution/checkpoints/div2k_gan/superres_epoch_0020.npz
    python training/super_resolution/upscale.py photos/ --checkpoint ... --output upscaled/
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from networks.autoencoders.super_resolution.model import SuperResolution
from networks.autoencoders.super_resolution.rrdb import RRDBNet

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def load_generator(path: Path, upscale: int) -> nn.Module:
    keys = np.load(path).files
    if any(key.startswith("conv_first") for key in keys):
        num_blocks = len({key.split(".")[2] for key in keys if key.startswith("trunk.layers.")})
        model = RRDBNet(upscale=upscale, num_blocks=num_blocks)
    else:
        model = SuperResolution(upscale=upscale, num_encoder_layers=5, latent_dim=256)
    model.load_weights(str(path), strict=True)
    model.train(False)
    return model


def upscale_image(
    model: nn.Module, image: np.ndarray, upscale: int, tile: int, overlap: int
) -> np.ndarray:
    """Upscale an RGB float32 [0, 1] image tile by tile; returns RGB float32 [0, 1]."""
    h, w, _ = image.shape
    out = np.zeros((h * upscale, w * upscale, 3), dtype=np.float32)
    for y in range(0, h, tile):
        for x in range(0, w, tile):
            # Run the model on the tile plus `overlap` pixels of context on each side...
            y0, x0 = max(y - overlap, 0), max(x - overlap, 0)
            y1, x1 = min(y + tile + overlap, h), min(x + tile + overlap, w)
            sr = mx.clip(model(mx.array(image[None, y0:y1, x0:x1])), 0.0, 1.0)
            sr = np.array(sr[0])
            # ...and keep only the output for the tile itself.
            th, tw = min(tile, h - y) * upscale, min(tile, w - x) * upscale
            sy, sx = (y - y0) * upscale, (x - x0) * upscale
            out[y * upscale : y * upscale + th, x * upscale : x * upscale + tw] = sr[
                sy : sy + th, sx : sx + tw
            ]
    return out


def collect_inputs(path: Path, upscale: int) -> list[Path]:
    if path.is_dir():
        # Skip earlier outputs, which land in the same directory by default.
        files = sorted(
            p
            for p in path.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS and not p.stem.endswith(f"_x{upscale}")
        )
        if not files:
            raise SystemExit(f"no images found in {path}")
        return files
    if not path.exists():
        raise SystemExit(f"{path} does not exist")
    return [path]


def output_path(source: Path, output: Path | None, upscale: int, many: bool) -> Path:
    name = f"{source.stem}_x{upscale}.png"
    if output is None:
        return source.with_name(name)
    if many or output.is_dir():
        output.mkdir(parents=True, exist_ok=True)
        return output / name
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Upscale images with a super-resolution model.")
    parser.add_argument("input", type=Path, help="An image, or a directory of images.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Generator .npz weights.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file or directory (default: <name>_x4.png next to each input).",
    )
    parser.add_argument("--upscale", type=int, default=4, help="Factor the model was trained for.")
    parser.add_argument("--tile", type=int, default=128, help="Tile size in input pixels.")
    parser.add_argument("--overlap", type=int, default=16, help="Context around each tile.")
    args = parser.parse_args()

    if not args.checkpoint.exists():
        raise SystemExit(f"checkpoint {args.checkpoint} does not exist")
    model = load_generator(args.checkpoint, args.upscale)

    inputs = collect_inputs(args.input, args.upscale)
    for source in inputs:
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"skipping {source}: could not read it as an image")
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        start = time.time()
        sr = upscale_image(model, rgb, args.upscale, args.tile, args.overlap)
        target = output_path(source, args.output, args.upscale, many=len(inputs) > 1)
        sr_bgr = cv2.cvtColor((sr * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(target), sr_bgr)

        h, w = rgb.shape[:2]
        print(
            f"{source} ({w}x{h}) -> {target} ({w * args.upscale}x{h * args.upscale}) "
            f"in {time.time() - start:.1f}s"
        )


if __name__ == "__main__":
    main()
