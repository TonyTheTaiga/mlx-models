import math

import mlx.core as mx


def get_tiles(image: mx.array, t: int = 16) -> tuple[mx.array, mx.array]:
    height, width, _ = image.shape
    num_tiles_x, num_tiles_y = math.ceil(width / t), math.ceil(height / t)

    tiles = []
    coords = []
    for j in range(num_tiles_y):
        ymin, ymax = j * t, min((j + 1) * t, height)
        for i in range(num_tiles_x):
            xmin, xmax = i * t, min((i + 1) * t, width)
            tiles.append(image[ymin:ymax, xmin:xmax, :])
            coords.append((xmin, xmax, ymin, ymax))

    return mx.stack(tiles), mx.array(coords, dtype=mx.float32)


def image_gradient_map(image: mx.array) -> mx.array:
    luminance_weights = mx.array([0.2126, 0.7152, 0.0722], dtype=mx.float32)
    grayscale = (image * luminance_weights).sum(axis=-1)
    padded = mx.pad(grayscale, [(1, 1), (1, 1)], mode="edge")
    grad_x = 0.5 * (padded[1:-1, 2:] - padded[1:-1, :-2])
    grad_y = 0.5 * (padded[2:, 1:-1] - padded[:-2, 1:-1])
    return mx.sqrt((grad_x * grad_x) + (grad_y * grad_y))


def build_prob_map(gradient_map: mx.array, l: float):
    if l < 0.0 or l > 1.0:
        raise ValueError("lambda needs to be [0, 1]")

    height, width = gradient_map.shape
    squared = gradient_map * gradient_map
    return (((1 - l) * squared) / mx.sum(squared)) + (l / (height * width))


__all__ = [
    "get_tiles",
    "image_gradient_map",
    "build_prob_map",
]
