import mlx.nn as nn
import mlx.core as mx


class Rope2D(nn.Module):
    def __init__(self, head_dim: int, n_rows: int, n_cols: int):
        super().__init__()

        assert head_dim % 4 == 0
        self.head_dim: int = head_dim
        self.pos_idx: mx.array = self._init_pos_index(n_rows=n_rows, n_cols=n_cols)

        j = mx.arange(head_dim / 4 - 1)

    def _init_pos_index(self, n_rows: int, n_cols: int) -> mx.array:
        pos = mx.arange(n_rows * n_cols)
        row = pos // n_cols
        col = pos % n_cols
        return mx.stack([row, col], axis=-1)

    def __call__(self, x: mx.array): ...


if __name__ == "__main__":
    j = mx.arange(32 / 4, dtype=mx.float32)
    freq = mx.pow(100.0, -j / (32 / 4))
    print(freq)
