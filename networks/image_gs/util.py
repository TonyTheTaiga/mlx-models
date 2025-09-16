import math

import mlx.core as mx

from .model import Gaussian2D


def get_tiles(image: mx.array, t: int = 16):
    """
    images are broken up into Ht x Wt and only gaussians whose minimum enclosing circle at 3std intersect a tile is that gaussian used to calculate the position in that tile.

    t = 16 in the paper
    """
    H, W, _ = image.shape
    nx, ny = math.ceil(W / t), math.ceil(H / t)

    tiles = []
    coords = []
    for j in range(ny):
        ymin, ymax = j * t, min((j + 1) * t, H)
        for i in range(nx):
            xmin, xmax = i * t, min((i + 1) * t, W)
            tiles.append(image[ymin:ymax, xmin:xmax, :])
            coords.append((xmin, xmax, ymin, ymax))

    return tiles, coords


def init_gaussians(image: mx.array) -> list[Gaussian2D]: ...


def image_gradient_map(image: mx.array) -> mx.array:
    # naive implementation - https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr
    w = mx.array([0.2126, 0.7152, 0.0722], dtype=mx.float32)
    gray = (image * w).sum(axis=-1)
    padded = mx.pad(gray, ((1, 1), (1, 1)), mode="edge")
    dx = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
    dy = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
    return mx.sqrt((dx * dx) + (dy * dy))
