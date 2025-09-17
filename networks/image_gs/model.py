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

    def render(self, target_idx: mx.array, precision_matrix: mx.array):
        assert self._tiles_and_coords is not None

        tiles, coords = self._tiles_and_coords  # (Nt, Th, Tw, 3), (Nt, 4) -> [xmin, xmax, ymin, ymax]

        Ng = self.mean.shape[0]
        Nt = tiles.shape[0]
        Th, Tw = int(tiles.shape[1]), int(tiles.shape[2])

        # for mahalanobis distance calc
        P11 = precision_matrix[:, 0, 0][:, None, None]  # (Ng, 1, 1)
        P12 = precision_matrix[:, 0, 1][:, None, None]  # (Ng, 1, 1) == P21
        P22 = precision_matrix[:, 1, 1][:, None, None]  # (Ng, 1, 1)

        # Outputs per tile
        out_tiles: list[mx.array] = []

        for j in range(Nt):
            # Tile pixel coordinates in absolute image space
            xmin, _, ymin, _ = coords[j, 0], coords[j, 1], coords[j, 2], coords[j, 3]
            xs = mx.arange(Tw, dtype=mx.float32) + xmin  # (Tw,)
            ys = mx.arange(Th, dtype=mx.float32) + ymin  # (Th,)

            XX = mx.broadcast_to(xs[None, :], (Th, Tw))  # (Th, Tw)
            YY = mx.broadcast_to(ys[:, None], (Th, Tw))  # (Th, Tw)

            # Differences to Gaussian means (broadcast over gaussians)
            dx = XX[None, :, :] - self.mean[:, 0][:, None, None]  # (Ng, Th, Tw)
            dy = YY[None, :, :] - self.mean[:, 1][:, None, None]  # (Ng, Th, Tw)

            # Mahalanobis distance using precision matrix Lambda:
            # q = [dx, dy]^T * Lambda * [dx, dy] = P11*dx^2 + 2*P12*dx*dy + P22*dy^2
            q = P11 * (dx * dx) + 2.0 * P12 * (dx * dy) + P22 * (dy * dy)  # (Ng, Th, Tw)
            w = mx.exp(-0.5 * q)  # unnormalized weight per Gaussian

            # Mask by tile/gaussian visibility (3σ circle test)
            mask_g = target_idx[:, j][:, None, None].astype(w.dtype)  # (Ng,1,1)
            w = w * mask_g  # zero out non-overlapping Gaussians

            # TODO: verify this for correctness
            k = min(self._top_k, Ng)
            w_sel = w.astype(mx.float32)
            jitter = (mx.arange(Ng, dtype=mx.float32) / float(Ng))[:, None, None] * 1e-6
            w_jit = w_sel + jitter
            top_vals = mx.topk(w_jit, k=k, axis=0)  # (k, Th, Tw)
            kth_val = mx.min(top_vals, axis=0)      # (Th, Tw) — threshold per pixel
            top_mask = (w_jit >= kth_val[None, :, :]).astype(w.dtype)

            # Normalize weights across selected Gaussians (avoid division by zero)
            w_top = w * top_mask
            denom = mx.sum(w_top, axis=0) + mx.array(1e-6, dtype=mx.float32)  # (Th, Tw)

            # Blend colors over the selected Gaussians
            rgb = mx.sum(w_top[:, :, :, None] * self.color[:, None, None, :], axis=0) / denom[:, :, None]
            out_tiles.append(rgb)

        return mx.stack(out_tiles, axis=0)

    def __call__(self, image: mx.array):
        if self._tiles_and_coords is None:
            self._tiles_and_coords = self._get_tiles(image=image)

        rotation_matrix = self._create_rotation_matrix(theta=mx.sigmoid(self.theta.astype(mx.float32)) * math.pi)
        eps = 1e-6
        inv_scale = nn.softplus(self.raw_inv_scale) + eps
        sigma = 1.0 / inv_scale
        precision_matrix = self._create_precision_matrix(rotation_matrix, inv_scale)

        radii = 3.0 * mx.maximum(sigma[:, 0], sigma[:, 1])
        target_idx = self._tile_hit(self._tiles_and_coords[1], self.mean, radii)
        rgb_tiled = self.render(target_idx, precision_matrix)

        H, W, _ = image.shape
        Th = int(rgb_tiled.shape[1])
        Tw = int(rgb_tiled.shape[2])
        Ny = math.ceil(H / self._tile_size)
        Nx = math.ceil(W / self._tile_size)

        grid = mx.reshape(rgb_tiled, (Ny, Nx, Th, Tw, 3))
        grid = mx.transpose(grid, (0, 2, 1, 3, 4))
        rgb = mx.reshape(grid, (Ny * Th, Nx * Tw, 3))
        H, W, _ = image.shape
        return rgb[:H, :W, :]
