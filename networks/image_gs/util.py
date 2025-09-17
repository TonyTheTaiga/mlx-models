import math

import mlx.core as mx


def get_tiles(image: mx.array, t: int = 16) -> tuple[mx.array, mx.array]:
    """
    Split the image into pixel-space tiles of size up to `t x t`.

    Images are broken up into Ht x Wt tiles (in pixel units) and only
    Gaussians whose minimum enclosing circle at 3σ intersects a tile are
    used to calculate contributions in that tile.

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

    # Note: `tiles` may be ragged at image borders when H or W is not a
    # multiple of `t`. The caller only uses `coords`; we keep the return
    # signature but recommend using the coords (pixel units).
    return mx.stack(tiles), mx.array(coords, dtype=mx.float32)


def image_gradient_map(image: mx.array) -> mx.array:
    # naive implementation - https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr
    w = mx.array([0.2126, 0.7152, 0.0722], dtype=mx.float32)
    gray = (image * w).sum(axis=-1)
    padded = mx.pad(gray, [(1, 1), (1, 1)], mode="edge")
    dx = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
    dy = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
    return mx.sqrt((dx * dx) + (dy * dy))


def build_prob_map(gradient_map: mx.array, l: float):
    if l < 0.0 or l > 1.0:
        raise ValueError("lambda needs to be [0, 1]")

    H, W = gradient_map.shape
    g2 = gradient_map * gradient_map
    return (((1 - l) * g2) / mx.sum(g2)) + (l / (H * W))
