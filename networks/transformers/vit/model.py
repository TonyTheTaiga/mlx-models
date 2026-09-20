import mlx.core as mx
import mlx.nn as nn

from networks.transformers.modules.attn import MultiHeadAttn


class Vit(nn.Module):
    def __init__(self, win_size: int):
        super().__init__()
        self.win_size: int = win_size

    def preprocess(self, x: mx.array) -> mx.array:
        if len(x.shape) == 3:
            x = x.reshape(1, *x.shape)

        return patchify(x, win_size=self.win_size)

    def forward(self, x: mx.array) -> mx.array: ...


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
    model = Vit(win_size=16)
    dummy = mx.ones((16, 320, 320, 3))
    patches = model.preprocess(dummy)
    print(patches.shape)
