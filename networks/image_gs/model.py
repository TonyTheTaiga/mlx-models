import math

import mlx.core as mx
import mlx.nn as nn

from .util import get_tiles


class ImageGS(nn.Module):
    def __init__(self, mean: mx.array, color: mx.array, tile_size: int, k: int = 10):
        super().__init__()

        self.mean = mean.astype(mx.float32)
        self.color = color.astype(mx.float32)
        self.theta = mx.zeros(shape=(mean.shape[0],), dtype=mx.float32)
        init_center = mx.array(-1.9, dtype=mx.float32)
        self.raw_inv_scale = init_center + (0.1 * mx.random.normal(shape=(mean.shape[0], 2), dtype=mx.float32))
        self._tile_size = tile_size
        self._tiles_and_coords: tuple[mx.array, mx.array] | None = None
        self._top_k = k

        self._init_center = init_center

    def _get_tiles(self, image: mx.array):
        return get_tiles(image, self._tile_size)

    def _create_rotation_matrix(self, theta: mx.array):
        cosine = mx.cos(theta)
        sine = mx.sin(theta)
        return mx.stack([mx.stack([cosine, -sine], axis=-1), mx.stack([sine, cosine], axis=-1)], axis=1)

    def _batched_diag(self, values: mx.array):
        diagonal = mx.zeros(shape=(values.shape[0], 2, 2), dtype=values.dtype)
        diagonal[:, 0, 0] = values[:, 0]
        diagonal[:, 1, 1] = values[:, 1]
        return diagonal

    def _create_precision_matrix(self, rotation_matrix: mx.array, inverse_scale: mx.array):
        diagonal_scale = self._batched_diag(inverse_scale * inverse_scale)
        return rotation_matrix @ diagonal_scale @ mx.transpose(rotation_matrix, (0, 2, 1))

    def _tile_hit(self, tiles: mx.array, centers: mx.array, radii: mx.array):
        x_min = tiles[:, 0][None, :]
        x_max = tiles[:, 1][None, :]
        y_min = tiles[:, 2][None, :]
        y_max = tiles[:, 3][None, :]
        center_x = centers[:, 0][:, None]
        center_y = centers[:, 1][:, None]
        radius = radii[:, None]
        clamped_x = mx.minimum(mx.maximum(center_x, x_min), x_max)
        clamped_y = mx.minimum(mx.maximum(center_y, y_min), y_max)
        delta_x = clamped_x - center_x
        delta_y = clamped_y - center_y
        return (delta_x * delta_x + delta_y * delta_y) <= (radius * radius)

    def render(self, target_idx: mx.array, precision_matrix: mx.array):
        assert self._tiles_and_coords is not None

        tiles, coords = self._tiles_and_coords

        num_gaussians = self.mean.shape[0]
        num_tiles = coords.shape[0]
        tile_height, tile_width = int(tiles.shape[1]), int(tiles.shape[2])

        precision_11 = precision_matrix[:, 0, 0][:, None, None]
        precision_12 = precision_matrix[:, 0, 1][:, None, None]
        precision_22 = precision_matrix[:, 1, 1][:, None, None]

        output_tiles: list[mx.array] = []

        for tile_index in range(num_tiles):
            x_min, _, y_min, _ = coords[tile_index, 0], coords[tile_index, 1], coords[tile_index, 2], coords[tile_index, 3]
            x_coords = mx.arange(tile_width, dtype=mx.float32) + x_min
            y_coords = mx.arange(tile_height, dtype=mx.float32) + y_min

            grid_x = mx.broadcast_to(x_coords[None, :], (tile_height, tile_width))
            grid_y = mx.broadcast_to(y_coords[:, None], (tile_height, tile_width))

            delta_x = grid_x[None, :, :] - self.mean[:, 0][:, None, None]
            delta_y = grid_y[None, :, :] - self.mean[:, 1][:, None, None]

            mahalanobis = precision_11 * (delta_x * delta_x) + 2.0 * precision_12 * (delta_x * delta_y) + precision_22 * (delta_y * delta_y)
            weights = mx.exp(-0.5 * mahalanobis)

            mask_visible = target_idx[:, tile_index][:, None, None].astype(weights.dtype)
            weights = weights * mask_visible

            k_top = min(self._top_k, num_gaussians)
            weights_selected = weights.astype(mx.float32)
            perturbation = (mx.arange(num_gaussians, dtype=mx.float32) / float(num_gaussians))[:, None, None] * 1e-6
            weights_perturbed = weights_selected + perturbation
            top_values = mx.topk(weights_perturbed, k=k_top, axis=0)
            kth_value = mx.min(top_values, axis=0)
            topk_mask = (weights_perturbed >= kth_value[None, :, :]).astype(weights.dtype)

            weights_topk = weights * topk_mask
            denominator = mx.sum(weights_topk, axis=0) + mx.array(1e-6, dtype=mx.float32)

            rgb_tile = mx.sum(weights_topk[:, :, :, None] * self.color[:, None, None, :], axis=0) / denominator[:, :, None]
            output_tiles.append(rgb_tile)

        return mx.stack(output_tiles, axis=0)

    def __call__(self, image: mx.array):
        if self._tiles_and_coords is None:
            self._tiles_and_coords = self._get_tiles(image=image)

        rotation_matrix = self._create_rotation_matrix(theta=mx.sigmoid(self.theta.astype(mx.float32)) * math.pi)
        epsilon = 1e-6
        inverse_scale = nn.softplus(self.raw_inv_scale) + epsilon
        standard_deviation = 1.0 / inverse_scale
        precision_matrix = self._create_precision_matrix(rotation_matrix, inverse_scale)

        influence_radius = 3.0 * mx.maximum(standard_deviation[:, 0], standard_deviation[:, 1])
        hit_mask = self._tile_hit(self._tiles_and_coords[1], self.mean, influence_radius)
        tiled_rgb = self.render(hit_mask, precision_matrix)

        height, width, _ = image.shape
        tile_height = int(tiled_rgb.shape[1])
        tile_width = int(tiled_rgb.shape[2])
        num_tiles_y = math.ceil(height / self._tile_size)
        num_tiles_x = math.ceil(width / self._tile_size)

        grid = mx.reshape(tiled_rgb, (num_tiles_y, num_tiles_x, tile_height, tile_width, 3))
        grid = mx.transpose(grid, (0, 2, 1, 3, 4))
        rgb_image = mx.reshape(grid, (num_tiles_y * tile_height, num_tiles_x * tile_width, 3))
        height, width, _ = image.shape
        return rgb_image[:height, :width, :]

    @property
    def num_gaussians(self) -> int:
        return int(self.mean.shape[0])

    def add_gaussians(self, mean: mx.array, color: mx.array):
        if mean.size == 0:
            return

        mean = mean.astype(mx.float32)
        color = color.astype(mx.float32)
        num_new = mean.shape[0]

        new_theta = mx.zeros(shape=(num_new,), dtype=mx.float32)
        new_raw_inv = self._init_center + (0.1 * mx.random.normal(shape=(num_new, 2), dtype=mx.float32))

        self.mean = mx.concatenate([self.mean, mean], axis=0)
        self.color = mx.concatenate([self.color, color], axis=0)
        self.theta = mx.concatenate([self.theta, new_theta], axis=0)
        self.raw_inv_scale = mx.concatenate([self.raw_inv_scale, new_raw_inv], axis=0)
