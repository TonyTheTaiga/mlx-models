import mlx.core as mx

from .model import ImageGS
from .util import image_gradient_map, build_prob_map


def build_model(
    image: mx.array, l: float = 0.5, num_initial_gaussians: int = 2000, tile_size: int = 16
) -> ImageGS:
    H, W, _ = image.shape
    gradient_map = image_gradient_map(image)
    pmf = build_prob_map(gradient_map, l)
    idx = mx.random.categorical(pmf.reshape(-1), num_samples=num_initial_gaussians)
    ys = idx // W
    xs = idx % W
    mean = mx.stack([xs, ys], axis=-1)
    color = image[ys, xs, :]
    return ImageGS(mean=mean, color=color, tile_size=tile_size)


__all__ = ["build_model"]
