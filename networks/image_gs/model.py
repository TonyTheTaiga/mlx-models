import math

import mlx.core as mx
import mlx.nn as nn

from .util import get_tiles


class ImageGS(nn.Module):
    def __init__(self, mean: mx.array, color: mx.array, tile_size: int):
        """Image Gaussian Splatting model operating strictly in pixel space.

        - `mean` must be pixel-center coordinates in `(x, y)` with
          `x in [0, W-1]` and `y in [0, H-1]`.
        - `color` is the RGB value sampled at those pixels.
        - All geometric calculations (means, scales, radii, tiles) are in pixels.
        """
        B, _ = mean.shape

        self.mean = mean.astype(mx.float16)
        self.color = color.astype(mx.float16)
        self.theta = mx.zeros(shape=(B,), dtype=mx.float16)
        self.raw_scale = mx.random.uniform(low=-3.0, high=3.0, shape=(B, 2), dtype=mx.float16)
        self._tile_size = tile_size
        self._tiles_and_coords = None

    def _get_tiles(self, image: mx.array):
        return get_tiles(image, self._tile_size)

    def _create_rotation_matrix(self, theta: mx.array):
        cos = mx.cos(theta)
        sin = mx.sin(theta)
        return mx.stack([mx.stack([cos, -sin], axis=-1), mx.stack([sin, cos], axis=-1)], axis=1)

    def _batched_diag(self, x: mx.array):
        diag = mx.zeros(shape=(x.shape[0], 2, 2), dtype=x.dtype)
        diag[:, 0, 0] = x[:, 0]
        diag[:, 1, 1] = x[:, 1]
        return diag

    def _create_precision_matrix(self, rotation_matrix: mx.array, inverse_scale: mx.array):
        diag_scale = self._batched_diag(inverse_scale * inverse_scale)
        return rotation_matrix @ diag_scale @ mx.transpose(rotation_matrix, (0, 2, 1))

    def _tile_hit(self, tiles: mx.array, centers: mx.array, radii: mx.array):
        xmin = tiles[:, 0][None, :]
        xmax = tiles[:, 1][None, :]
        ymin = tiles[:, 2][None, :]
        ymax = tiles[:, 3][None, :]
        cx = centers[:, 0][:, None]
        cy = centers[:, 1][:, None]
        r = radii[:, None]
        clamped_x = mx.minimum(mx.maximum(cx, xmin), xmax)
        clamped_y = mx.minimum(mx.maximum(cy, ymin), ymax)
        dx = clamped_x - cx
        dy = clamped_y - cy
        return (dx * dx + dy * dy) <= (r * r)

    def __call__(self, image: mx.array):
        if self._tiles_and_coords is None:
            self._tiles_and_coords = self._get_tiles(image=image)

        rotation_matrix = self._create_rotation_matrix(theta=mx.sigmoid(self.theta) * math.pi)
        eps = 1e-6
        sigma = nn.softplus(self.raw_scale) + eps

        inv_scale = 1.0 / sigma
        precision_matrix = self._create_precision_matrix(rotation_matrix, inv_scale)

        radii = 3.0 * mx.maximum(sigma[:, 0], sigma[:, 1])
        target_idx = self._tile_hit(self._tiles_and_coords[1], self.mean, radii)
