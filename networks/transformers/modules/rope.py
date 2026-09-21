import mlx.nn as nn
import mlx.core as mx
from numpy import arange


class Rope2D(nn.Module):
    def __init__(self, head_dim: int, n_rows: int, n_cols: int, base: int):
        super().__init__()

        assert head_dim % 4 == 0
        self.head_dim: int = head_dim
        self.pos_idx: mx.array = self._init_pos_index(n_rows=n_rows, n_cols=n_cols)

        n_freqs = head_dim // 4
        j = mx.arange(n_freqs, dtype=mx.float32)
        self.freq: mx.array = mx.pow(base, -j / n_freqs)

    def _init_pos_index(self, n_rows: int, n_cols: int) -> mx.array:
        pos = mx.arange(n_rows * n_cols)
        row = pos // n_cols
        col = pos % n_cols
        return mx.stack([row, col], axis=-1)

    def __call__(self, x: mx.array):
        """
        x (mx.array): (b, n_heads, num_patches, features per head)
        """
        rows = self.pos_idx[:, 0]
        cols = self.pos_idx[:, 1]
        radians = mx.concat(
            [rows[:, None] * self.freq[None, :], cols[:, None] * self.freq[None, :]],
            axis=-1,
        )
        s = mx.sin(radians)[None, None, :]
        c = mx.cos(radians)[None, None, :]
        batch, n_heads, n_patches, n_dims = x.shape
        x_paired: mx.array = x.reshape(batch, n_heads, n_patches, n_dims // 2, 2)
        return mx.stack(
            [
                x_paired[..., 0] * c - x_paired[..., 1] * s,
                x_paired[..., 0] * s + x_paired[..., 1] * c,
            ],
            axis=-1,
        ).reshape(batch, n_heads, n_patches, n_dims)


if __name__ == "__main__":
    n = mx.ones((1, 49, 16))  # Nonzero pairs expose incorrect rotations.
    n_heads = 2
    rope = Rope2D(head_dim=8, n_rows=7, n_cols=7, base=100)
    rotated = (
        rope(n.reshape(1, 49, n_heads, 16 // n_heads).transpose(0, 2, 1, 3))
        .transpose(0, 2, 1, 3)
        .reshape(1, 49, 16)
    )
    assert rotated.shape == n.shape
    before = mx.sum(n.reshape(1, 49, 8, 2) ** 2, axis=-1)
    after = mx.sum(rotated.reshape(1, 49, 8, 2) ** 2, axis=-1)
    assert mx.allclose(before, after).item()  # Rotation preserves each pair's length.
    assert not mx.allclose(n[:, 1:], rotated[:, 1:]).item()
