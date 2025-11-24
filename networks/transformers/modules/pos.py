import math

import mlx.core as mx
import mlx.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()

        pos = mx.arange(0, max_seq_len, dtype=mx.float32)
        pos = mx.expand_dims(pos, 1)
        div = mx.exp(mx.arange(0, d_model, 2, dtype=mx.float32) * -math.log(10000) / d_model)
        self._pe = mx.zeros(shape=(max_seq_len, d_model))
        self._pe[:, 0::2] = mx.sin(pos * div)
        self._pe[:, 1::2] = mx.cos(pos * div)
        self._pe = mx.expand_dims(self._pe, 0)

    def __call__(self, x: mx.array, current_position: int | None = None):
        if current_position is not None:
            return x + self._pe[:, current_position]

        return x + self._pe[:, : x.shape[1]]
