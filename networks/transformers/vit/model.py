from typing import final, override

import mlx.core as mx
import mlx.nn as nn

from networks.transformers.modules.rope import Rope2D


class EncoderBlock(nn.Module):
    def __init__(self, n_heads: int, d_model: int):
        super().__init__()

        self.proj = nn.Linear(input_dims=d_model, output_dims=3 * d_model, bias=False)
        self.n_heads = n_heads
        self.d_model = d_model

        self.pn1 = nn.LayerNorm(dims=d_model)
        self.pn2 = nn.LayerNorm(dims=d_model)

        self.roper = Rope2D(head_dim=d_model // n_heads, n_rows=4, n_cols=4, base=100)

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
        return x.reshape(
            batch, seqlen, self.n_heads, self.d_model // self.n_heads
        ).transpose(0, 2, 1, 3)

    @final
    @override
    def __call__(self, x: mx.array) -> mx.array:
        y = self.proj(self.pn1(x))
        q, k, v = mx.split(y, 3, axis=-1)
        q = self.roper(self.reshape(q))
        k = self.roper(self.reshape(k))
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


class Vit(nn.Module):
    def __init__(
        self,
        win_size: int,
        n_layers: int,
        patch_features: int,
        d_model: int,
        n_heads: int,
    ):
        super().__init__()
        self.win_size: int = win_size

        self.proj = nn.Linear(
            input_dims=patch_features, output_dims=d_model, bias=False
        )

        self.encoder_stack = nn.Sequential(
            *[EncoderBlock(n_heads=n_heads, d_model=d_model) for _ in range(n_layers)]
        )

    def preprocess(self, x: mx.array) -> mx.array:
        if len(x.shape) == 3:
            x = x.reshape(1, *x.shape)

        return patchify(x, win_size=self.win_size)

    @final
    @override
    def __call__(self, x: mx.array) -> mx.array:
        return self.encoder_stack(self.proj(x))


def patchify(x: mx.array, win_size: int) -> mx.array:
    assert (
        x.shape[1] % win_size == 0 and x.shape[2] % win_size == 0
    ), "h, w needs to be cleanly divisible by win_size"

    def _patch_item(_input: mx.array) -> mx.array:
        _patches: list[mx.array] = []
        for _y in range(0, _input.shape[0], win_size):
            for _x in range(0, _input.shape[1], win_size):
                _patches.append(
                    _input[_y : _y + win_size, _x : _x + win_size].flatten()
                )

        return mx.stack(_patches)

    patches: list[mx.array] = []
    for item in x:
        patches.append(_patch_item(item))

    return mx.stack(patches)


if __name__ == "__main__":
    # q = mx.random.normal((3, 2, 100, 16))
    # k = mx.random.normal((3, 2, 100, 16))
    # v = mx.random.normal((3, 2, 100, 16))

    # x = mx.fast.scaled_dot_product_attention(q, k, v, scale=16)
    # print(x.shape)

    model = Vit(win_size=7, n_layers=4, patch_features=49, d_model=32, n_heads=1)
    _input = model.preprocess(mx.random.normal(shape=(2, 28, 28, 1)))
    mx.eval(model(_input))
    mx.metal.start_capture("vit.gputrace")
    mx.eval(model(_input))
    mx.metal.stop_capture()
