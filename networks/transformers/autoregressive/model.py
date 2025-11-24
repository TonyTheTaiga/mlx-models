import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from networks.transformers.modules.attn import MultiHeadAttn
from networks.transformers.modules.pos import PositionalEncoding


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()

        self.attn = MultiHeadAttn(d_model=d_model, n_heads=n_heads)
        self.ln1 = nn.RMSNorm(d_model)

        self.ff = nn.Sequential(
            *[nn.Linear(d_model, d_ff, bias=False), nn.ReLU(), nn.Linear(d_ff, d_model, bias=False)]
        )
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

        if max_gen <= 0:
            return

        max_steps = min(max_gen, self._seq_len - x.shape[1])
        if max_steps <= 0:
            raise ValueError(
                "Cannot generate any tokens because the sequence is already at max length"
            )

        cache = []
        mask = MultiHeadAttn.generate_causal_mask(x.shape[1])
        mask = mask.astype(self.token_embedding.weight.dtype)

        x = self.token_embedding(x) * math.sqrt(self._d_model)
        hidden_states = self.positional_encoding(x)
        for layer in self.layers:
            hidden_states, cache_layer = layer(hidden_states, mask=mask)
            cache.append(cache_layer)

        current_position = x.shape[1]
        for _ in range(max_steps):
            logits = self.output_layer(self.ln1(hidden_states)[:, -1])
            next_token = mx.random.categorical(logits * (1 / temperature))
            yield next_token
            current_position += 1

            if current_position >= self._seq_len:
                break

            next_token = next_token[:, None]
            token_emb = self.token_embedding(next_token) * math.sqrt(self._d_model)
            hidden_states = self.positional_encoding(
                token_emb, current_position=current_position - 1
            )

            for i, layer in enumerate(self.layers):
                hidden_states, cache[i] = layer(hidden_states, kv_cache=cache[i])


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
