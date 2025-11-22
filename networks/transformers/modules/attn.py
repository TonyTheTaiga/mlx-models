import math

import mlx.core as mx
import mlx.nn as nn


class MultiHeadAttn(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        assert d_model % n_heads == 0, "d_model should evenly divide by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def generate_causal_mask(seq_len: int):
        mask = mx.triu(mx.ones(shape=(seq_len, seq_len), dtype=mx.float32), k=1)
        mask = mask * -1e9
        return mask

    def scaled_dot_product(
        self, Q: mx.array, K: mx.array, V: mx.array, mask: mx.array | None = None
    ):
        scores = (Q @ K.transpose(0, 1, 3, 2)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores + mask

        scores = mx.softmax(scores, -1)
        values = (scores @ V).transpose(0, 2, 1, 3)
        return values, scores

    def __call__(
        self,
        x: mx.array,
        kv_cache: tuple[mx.array, mx.array] | None = None,
        mask: mx.array | None = None,
    ):
        b, slen, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.reshape(b, slen, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(b, slen, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(b, slen, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

        if kv_cache is not None:
            cached_K, cached_V = kv_cache
            K = mx.concatenate([cached_K, K], axis=2)
            V = mx.concatenate([cached_V, V], axis=2)

        values, _ = self.scaled_dot_product(Q, K, V, mask)
        values = values.reshape(b, slen, self.n_heads * self.d_k)
        out = self.W_o(values)
        return out, (K, V)
