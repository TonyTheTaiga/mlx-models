import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

mx.random.seed(420)


class MultiHeadAttn(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        assert d_model % n_heads == 0, "d_model must be be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    @staticmethod
    def generate_causal_mask(seq_len: int):
        mask = mx.triu(mx.ones(shape=(seq_len, seq_len), dtype=mx.float32), k=1)
        mask = mask * -1e9
        return mask

    def scaled_dot_product(
        self, Q: mx.array, K: mx.array, V: mx.array, mask: Optional[mx.array] = None
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
        kv_cache: Optional[tuple[mx.array, mx.array]] = None,
        mask: Optional[mx.array] = None,
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

    def __call__(self, x: mx.array, current_position: Optional[int] = None):
        if current_position is not None:
            return x + self._pe[:, current_position]

        return x + self._pe[:, : x.shape[1]]


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.l1 = nn.Linear(d_model, d_ff, bias=False)
        self.l2 = nn.Linear(d_ff, d_model, bias=False)
        self.relu = nn.ReLU()

    def __call__(self, x):
        return self.l2(self.relu(self.l1(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()

        self.attn = MultiHeadAttn(d_model=d_model, n_heads=n_heads)
        self.ln1 = nn.RMSNorm(d_model)

        self.ff = FeedForward(d_model, d_ff)
        self.ln2 = nn.RMSNorm(d_model)

    def __call__(
        self,
        x: mx.array,
        kv_cache: Optional[tuple[mx.array, mx.array]] = None,
        mask: Optional[mx.array] = None,
    ):
        attn_weights, kv_cache = self.attn(self.ln1(x), kv_cache=kv_cache, mask=mask)
        x = attn_weights + x
        return self.ff(self.ln2(x)) + x, kv_cache


class Transformer(nn.Module):
    def __init__(
        self, vocab_size: int, seq_len: int, d_model: int, d_ff: int, n_heads: int, n_layers: int
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, seq_len)
        self.layers = [DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln1 = nn.RMSNorm(d_model)
        self.output_layer = nn.Linear(d_model, vocab_size, bias=False)

        # self.output_layer.weight = self.token_embedding.weight

        self._seq_len = seq_len
        self._d_model = d_model

    def __call__(self, x):
        x = self.positional_encoding(self.token_embedding(x) * math.sqrt(self._d_model))

        mask = MultiHeadAttn.generate_causal_mask(x.shape[1])
        mask = mask.astype(self.token_embedding.weight.dtype)
        for layer in self.layers:
            x, _ = layer(x, kv_cache=None, mask=mask)

        x = self.ln1(x)
        y = self.output_layer(x)
        return y

    def generate(self, x, max_gen: int = 100, temperature: float = 1.0):
        if x.shape[1] > self._seq_len:
            raise ValueError(
                f"Initial sequence length {x.shape[1]} exceeds maximum allowed length {self._seq_len}"
            )

        cache = []
        mask = MultiHeadAttn.generate_causal_mask(x.shape[1])
        mask = mask.astype(self.token_embedding.weight.dtype)

        current_position = x.shape[1] - 1
        x = self.token_embedding(x) * math.sqrt(self._d_model)
        hidden_states = self.positional_encoding(x)
        for layer in self.layers:
            hidden_states, cache_layer = layer(hidden_states, mask=mask)
            cache.append(cache_layer)

        logits = self.output_layer(self.ln1(hidden_states)[:, -1])
        next_token = mx.random.categorical(logits * (1 / temperature))
        yield next_token
        current_position += 1

        while current_position < max_gen:
            next_token = next_token[:, None]
            token_emb = self.token_embedding(next_token) * math.sqrt(self._d_model)
            hidden_states = self.positional_encoding(token_emb, current_position=current_position)

            for i, layer in enumerate(self.layers):
                hidden_states, cache[i] = layer(hidden_states, kv_cache=cache[i])

            logits = self.output_layer(self.ln1(hidden_states)[:, -1])
            next_token = mx.random.categorical(logits * (1 / temperature))
            yield next_token
            current_position += 1


if __name__ == "__main__":
    d_model = 4
    d_ff = 3072
    max_seq_len = 8
    vocab_size = 10
    n_heads = 1
    n_layers = 6

    model = Transformer(vocab_size, max_seq_len, d_model, d_ff, n_heads, n_layers)
    x = mx.random.randint(1, vocab_size, shape=(1, max_seq_len))
    model(x)
