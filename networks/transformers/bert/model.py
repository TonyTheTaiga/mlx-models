import mlx.core as mx
import mlx.nn as nn

from networks.transformers.modules.attn import MultiHeadAttn
from networks.transformers.modules.pos import PositionalEncoding


class Transformer(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()

        self.multi_head_attn = MultiHeadAttn(d_model=d_model, n_heads=n_heads)
        self.norm_1 = nn.LayerNorm(dims=d_model)
        self.ffn = nn.Sequential(
            *[
                nn.Linear(input_dims=d_model, output_dims=d_model * 4, bias=True),
                nn.GELU(),
                nn.Linear(input_dims=d_model * 4, output_dims=d_model, bias=True),
            ]
        )
        self.norm_2 = nn.LayerNorm(dims=d_model)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = self.norm_1(self.multi_head_attn(x, mask=mask)[0] + x)
        return self.norm_2(self.ffn(x) + x)


class Bert(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        n_layers: int,
    ) -> None:
        super().__init__()

        self.embedder = nn.Embedding(vocab_size, d_model)
        self.position_encoder = PositionalEncoding(d_model=d_model, max_seq_len=max_seq_len)
        self.layers = [Transformer(d_model, n_heads) for _ in range(n_layers)]

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = self.position_encoder(self.embedder(x))
        for layer in self.layers:
            x = layer(x, mask)

        return x


if __name__ == "__main__":
    vocab_size = 20
    seq_len = 8
    d_model = 32
    n_heads = 4
    n_layers = 2
    pad_id = 0

    model = Bert(
        vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        max_seq_len=seq_len,
        n_layers=n_layers,
    )

    # A toy batch with padding on the right.
    tokens = mx.array([[5, 7, 3, 9, 0, 0, 0, 0]], dtype=mx.int32)

    # Build an additive attention mask: 0 for valid tokens, -inf for padding.
    attention_mask = tokens != pad_id
    attn_mask = attention_mask[:, None, :, None] * attention_mask[:, None, None, :]  # pyright: ignore
    attn_mask = mx.where(attn_mask, 0.0, -1e9).astype(mx.float32)

    encoded = model(tokens, mask=attn_mask)
    print("Encoded shape:", encoded.shape)
    print("First token embedding (truncated):")
    print(encoded[0, 0, :8])
