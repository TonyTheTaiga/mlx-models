import mlx.core as mx

from .model import ImageGS
from .util import image_gradient_map, build_prob_map


def build_model(
    image: mx.array, l: float = 0.5, samples: int = 2000, tile_size: int = 16
) -> ImageGS:
    """Construct an ImageGS by sampling points using a gradient-based PMF.

    - `l` blends gradient magnitude with uniform sampling (0..1).
    - `samples` controls the number of Gaussians (points) to sample.
    - `tile_size` is forwarded to the ImageGS initializer.
    - Coordinates are in pixel space: `(x, y)` with `x in [0, W-1]`,
      `y in [0, H-1]`.
    """
    H, W, _ = image.shape
    gradient_map = image_gradient_map(image)
    pmf = build_prob_map(gradient_map, l)
    idx = mx.random.categorical(pmf.reshape(-1), num_samples=samples)
    ys = idx // W
    xs = idx % W
    mean = mx.stack([(xs).astype(mx.float32), (ys).astype(mx.float32)], axis=-1)
    color = image[ys, xs, :]
    return ImageGS(mean=mean, color=color, tile_size=tile_size)


__all__ = ["build_model"]
