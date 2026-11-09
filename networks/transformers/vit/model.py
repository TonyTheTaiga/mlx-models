from typing import final, override

import mlx.core as mx
import mlx.nn as nn

from networks.transformers.modules.rope import Rope2D


class EncoderBlock(nn.Module):
    def __init__(self, n_heads: int, d_model: int, n_rows: int, n_cols: int):
        super().__init__()

        self.proj = nn.Linear(input_dims=d_model, output_dims=3 * d_model, bias=False)
        self.n_heads = n_heads
        self.d_model = d_model

        self.pn1 = nn.LayerNorm(dims=d_model)
        self.pn2 = nn.LayerNorm(dims=d_model)

        self.roper = Rope2D(head_dim=d_model // n_heads, n_rows=n_rows, n_cols=n_cols, base=100)

        self.feed_forward = nn.Sequential(
            *[
                nn.Linear(input_dims=d_model, output_dims=d_model * 4, bias=False),
                nn.GELU(),
                nn.Linear(input_dims=d_model * 4, output_dims=d_model, bias=False),
            ]
        )

        self.atten_proj = nn.Linear(input_dims=d_model, output_dims=d_model, bias=False)

    def reshape(self, x: mx.array) -> mx.array:
        # input: [batch, seqlen, d_model]
        # output: [batch, n_heads, seqlen, d_model // n_heads]
        batch, seqlen, _ = x.shape
        return x.reshape(batch, seqlen, self.n_heads, self.d_model // self.n_heads).transpose(
            0, 2, 1, 3
        )

    @final
    @override
    def __call__(self, x: mx.array) -> mx.array:
        y = self.proj(self.pn1(x))
        q, k, v = mx.split(y, 3, axis=-1)
        q = self.apply_rope(self.reshape(q))
        k = self.apply_rope(self.reshape(k))
        v = self.reshape(v)
        attention = (
            mx.fast.scaled_dot_product_attention(
                q, k, v, scale=(self.d_model // self.n_heads) ** -0.5
            )
            .transpose(0, 2, 1, 3)
            .reshape(*x.shape)
        )
        x = x + self.atten_proj(attention)
        z = self.feed_forward(self.pn2(x))
        return x + z

    def apply_rope(self, x: mx.array) -> mx.array:
        n_patches = self.roper.pos_idx.shape[0]
        if x.shape[2] == n_patches + 1:
            return mx.concatenate([x[:, :, :1], self.roper(x[:, :, 1:])], axis=2)
        return self.roper(x)


class Vit(nn.Module):
    def __init__(
        self,
        win_size: int,
        n_layers: int,
        d_model: int,
        n_heads: int,
        image_size: int,
        channels: int,
    ):
        super().__init__()
        assert image_size % win_size == 0, "image_size needs to be cleanly divisible by win_size"
        self.win_size: int = win_size
        grid = image_size // win_size

        self.proj = nn.Linear(
            input_dims=win_size * win_size * channels, output_dims=d_model, bias=False
        )

        self.encoder_stack = nn.Sequential(
            *[
                EncoderBlock(n_heads=n_heads, d_model=d_model, n_rows=grid, n_cols=grid)
                for _ in range(n_layers)
            ]
        )

    def preprocess(self, x: mx.array) -> mx.array:
        if len(x.shape) == 3:
            x = x.reshape(1, *x.shape)

        return patchify(x, win_size=self.win_size)

    @final
    @override
    def __call__(self, x: mx.array, prefix_tokens: mx.array | None = None) -> mx.array:
        patch_embeddings = self.proj(x)
        if prefix_tokens is not None:
            patch_embeddings = mx.concatenate([prefix_tokens, patch_embeddings], axis=1)
        return self.encoder_stack(patch_embeddings)


def patchify(x: mx.array, win_size: int) -> mx.array:
    """
    x (mx.array): (b, h, w, c) -> (b, num_patches, win_size * win_size * c), patches row-major
    """
    batch, h, w, c = x.shape
    assert h % win_size == 0 and w % win_size == 0, "h, w needs to be cleanly divisible by win_size"

    rows, cols = h // win_size, w // win_size
    return (
        x.reshape(batch, rows, win_size, cols, win_size, c)
        .transpose(0, 1, 3, 2, 4, 5)
        .reshape(batch, rows * cols, win_size * win_size * c)
    )


if __name__ == "__main__":
    # q = mx.random.normal((3, 2, 100, 16))
    # k = mx.random.normal((3, 2, 100, 16))
    # v = mx.random.normal((3, 2, 100, 16))

    # x = mx.fast.scaled_dot_product_attention(q, k, v, scale=16)
    # print(x.shape)

    model = Vit(win_size=7, n_layers=4, d_model=32, n_heads=1, image_size=28, channels=1)
    _input = model.preprocess(mx.random.normal(shape=(2, 28, 28, 1)))
    mx.eval(model(_input))
    mx.metal.start_capture("vit.gputrace")
    mx.eval(model(_input))
    mx.metal.stop_capture()
