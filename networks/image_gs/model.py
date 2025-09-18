import math

import mlx.core as mx
import mlx.nn as nn

from .util import get_tiles


class ImageGS(nn.Module):
    def __init__(self, mean: mx.array, color: mx.array, tile_size: int, k: int = 10):
        super().__init__()

        self.mean = mean.astype(mx.float16)
        self.color = color.astype(mx.float16)
        self.theta = mx.zeros(shape=(mean.shape[0],), dtype=mx.float16)
        self.inv_scale = mx.full(shape=(mean.shape[0], 2), vals=0.2, dtype=mx.float16)
        self._tile_size = tile_size
        self._tiles_and_coords: tuple[mx.array, mx.array] | None = None
        self._top_k = k

    def _get_tiles(self, image: mx.array):
        return get_tiles(image, self._tile_size)

    def _create_rotation_matrix(self, theta: mx.array):
        cosine = mx.cos(theta)
        sine = mx.sin(theta)
        return mx.stack(
            [mx.stack([cosine, -sine], axis=-1), mx.stack([sine, cosine], axis=-1)], axis=1
        )

    def _batched_diag(self, values: mx.array):
        diagonal = mx.zeros(shape=(values.shape[0], 2, 2), dtype=values.dtype)
        diagonal[:, 0, 0] = values[:, 0]
        diagonal[:, 1, 1] = values[:, 1]
        return diagonal

    def _create_precision_matrix(self, rotation_matrix: mx.array, inverse_scale: mx.array):
        diagonal_scale = self._batched_diag(inverse_scale * inverse_scale)
        return rotation_matrix @ diagonal_scale @ mx.transpose(rotation_matrix, (0, 2, 1))

    def _tile_hit(self, tile_bounds: mx.array, centers_px: mx.array, radii_px: mx.array):
        x_min = tile_bounds[:, 0][None, :]
        x_max = tile_bounds[:, 1][None, :]
        y_min = tile_bounds[:, 2][None, :]
        y_max = tile_bounds[:, 3][None, :]
        center_x = centers_px[:, 0][:, None]
        center_y = centers_px[:, 1][:, None]
        radius = radii_px[:, None]
        clamped_x = mx.minimum(mx.maximum(center_x, x_min), x_max)
        clamped_y = mx.minimum(mx.maximum(center_y, y_min), y_max)
        dx = clamped_x - center_x
        dy = clamped_y - center_y
        return (dx * dx + dy * dy) <= (radius * radius)

    def render(self, tile_visibility_mask: mx.array, mean_px: mx.array, precision_matrix: mx.array):
        assert self._tiles_and_coords is not None
        tile_images, tile_bounds = self._tiles_and_coords
        mean_f32 = mean_px.astype(mx.float32)
        color_f32 = self.color.astype(mx.float32)
        num_gaussians = self.mean.shape[0]
        num_tiles = tile_bounds.shape[0]
        tile_height, tile_width = int(tile_images.shape[1]), int(tile_images.shape[2])
        precision_11 = precision_matrix[:, 0, 0][:, None, None]
        precision_12 = precision_matrix[:, 0, 1][:, None, None]
        precision_22 = precision_matrix[:, 1, 1][:, None, None]
        rendered_tiles: list[mx.array] = []
        for tile_index in range(num_tiles):
            x_min, _, y_min, _ = (
                tile_bounds[tile_index, 0],
                tile_bounds[tile_index, 1],
                tile_bounds[tile_index, 2],
                tile_bounds[tile_index, 3],
            )
            x_coords = mx.arange(tile_width, dtype=mx.float32) + x_min
            y_coords = mx.arange(tile_height, dtype=mx.float32) + y_min
            pixel_x_grid = mx.broadcast_to(x_coords[None, :], (tile_height, tile_width))
            pixel_y_grid = mx.broadcast_to(y_coords[:, None], (tile_height, tile_width))
            dx = pixel_x_grid[None, :, :] - mean_f32[:, 0][:, None, None]
            dy = pixel_y_grid[None, :, :] - mean_f32[:, 1][:, None, None]
            mahalanobis = (
                precision_11 * (dx * dx) + 2.0 * precision_12 * (dx * dy) + precision_22 * (dy * dy)
            )
            weights = mx.exp(-0.5 * mahalanobis)
            visible = tile_visibility_mask[:, tile_index][:, None, None].astype(weights.dtype)
            weights = weights * visible
            top_k = min(self._top_k, num_gaussians)
            weights32 = weights.astype(mx.float32)
            perturb = (mx.arange(num_gaussians, dtype=mx.float32) / float(num_gaussians))[
                :, None, None
            ] * 1e-6
            weights32 = weights32 + perturb
            top_values = mx.topk(weights32, k=top_k, axis=0)
            kth_value = mx.min(top_values, axis=0)
            topk_mask = (weights32 >= kth_value[None, :, :]).astype(weights.dtype)
            weights_topk = weights * topk_mask
            denom = mx.sum(weights_topk, axis=0) + mx.array(1e-6, dtype=mx.float32)
            tile_rgb = (
                mx.sum(weights_topk[:, :, :, None] * color_f32[:, None, None, :], axis=0)
                / denom[:, :, None]
            )
            rendered_tiles.append(tile_rgb)
        return mx.stack(rendered_tiles, axis=0)

    def __call__(self, image: mx.array):
        if self._tiles_and_coords is None:
            self._tiles_and_coords = self._get_tiles(image=image)

        height, width, _ = image.shape
        theta_mapped = (mx.sigmoid(self.theta.astype(mx.float32)) * 2.0 - 1.0) * (math.pi / 2.0)
        rotation_matrix = self._create_rotation_matrix(theta=theta_mapped)
        inverse_scale = self.inv_scale.astype(mx.float32)
        standard_deviation = 1.0 / inverse_scale
        precision_matrix = self._create_precision_matrix(rotation_matrix, inverse_scale)

        influence_radius = 3.0 * mx.maximum(standard_deviation[:, 0], standard_deviation[:, 1])
        mean_px = mx.stack(
            [
                self.mean.astype(mx.float32)[:, 0] * mx.array(width, dtype=mx.float32),
                self.mean.astype(mx.float32)[:, 1] * mx.array(height, dtype=mx.float32),
            ],
            axis=-1,
        )
        hit_mask = self._tile_hit(self._tiles_and_coords[1], mean_px, influence_radius)
        tiled_rgb = self.render(hit_mask, mean_px, precision_matrix)

        tile_height = int(tiled_rgb.shape[1])
        tile_width = int(tiled_rgb.shape[2])
        num_tiles_y = math.ceil(height / self._tile_size)
        num_tiles_x = math.ceil(width / self._tile_size)

        grid = mx.reshape(tiled_rgb, (num_tiles_y, num_tiles_x, tile_height, tile_width, 3))
        grid = mx.transpose(grid, (0, 2, 1, 3, 4))
        rgb_image = mx.reshape(grid, (num_tiles_y * tile_height, num_tiles_x * tile_width, 3))
        return rgb_image[:height, :width, :]

    @property
    def num_gaussians(self) -> int:
        return int(self.mean.shape[0])

    def add_gaussians(self, mean: mx.array, color: mx.array):
        if mean.size == 0:
            return

        mean = mean.astype(mx.float16)
        color = color.astype(mx.float16)
        num_new = mean.shape[0]

        new_theta = mx.zeros(shape=(num_new,), dtype=mx.float16)
        new_inv = mx.full(shape=(num_new, 2), vals=0.2, dtype=mx.float16)

        self.mean = mx.concatenate([self.mean, mean], axis=0)
        self.color = mx.concatenate([self.color, color], axis=0)
        self.theta = mx.concatenate([self.theta, new_theta], axis=0)
        self.inv_scale = mx.concatenate([self.inv_scale, new_inv], axis=0)
