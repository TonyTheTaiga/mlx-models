import mlx.core as mx

from .model import ImageGS
from .util import image_gradient_map, build_prob_map


def build_model(
    image: mx.array, l: float = 0.5, num_initial_gaussians: int = 2000, tile_size: int = 16
) -> ImageGS:
    height, width, _ = image.shape
    gradient_map = image_gradient_map(image)
    probability_mass = build_prob_map(gradient_map, l)
    sampled_indices = mx.random.categorical(probability_mass.reshape(-1), num_samples=num_initial_gaussians)
    rows = sampled_indices // width
    cols = sampled_indices % width
    means = mx.stack([cols, rows], axis=-1)
    colors = image[rows, cols, :]
    return ImageGS(mean=means, color=colors, tile_size=tile_size)


__all__ = ["build_model"]
