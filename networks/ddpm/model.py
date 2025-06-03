import math

import mlx.core as mx
import mlx.nn as nn


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, max_t: int):
        super().__init__()

        self._te = mx.zeros(shape=(max_t, dim))
        pos = mx.arange(0, max_t, dtype=mx.float32)
        pos = mx.expand_dims(pos, 1)
        div = mx.exp(mx.arange(0, dim, 2, dtype=mx.float32) * (-math.log(10000) / dim))
        self._te[:, 0::2] = mx.sin(pos * div)
        self._te[:, 1::2] = mx.cos(pos * div)

    def __call__(self, t):
        # try adding mlp here?
        return self._te[t]


class FiLM(nn.Module):
    def __init__(self, idim, odim):
        super().__init__()

        hdim = 4 * idim
        self.h1 = nn.Linear(idim, hdim)
        self.h2 = nn.Linear(hdim, 2 * odim)
        self.h2.weight[:] = 0
        self.h2.bias[:] = 0

    def __call__(self, x):
        x = self.h1(x)
        x = nn.silu(x)
        x = self.h2(x)
        gamma, beta = mx.split(x, 2, axis=-1)
        return gamma, beta


class Block(nn.Module):
    def __init__(self, idim, odim, embed_dim=128):
        super().__init__()

        self.identity = nn.Identity() if idim == odim else nn.Conv2d(idim, odim, 1)
        self.cn1 = nn.Conv2d(idim, odim, kernel_size=3, stride=1, padding=1)
        self.gn1 = nn.GroupNorm(8, odim)
        self.cn2 = nn.Conv2d(odim, odim, kernel_size=3, stride=1, padding=1)
        self.gn2 = nn.GroupNorm(8, odim)
        self.film = FiLM(embed_dim, odim)

    def __call__(self, x: mx.array, time_embedding: mx.array) -> mx.array:
        gamma, beta = self.film(time_embedding)

        x_1 = self.gn1(self.cn1(x))
        x_1 = (1 + gamma)[:, None, None, :] * x_1 + beta[:, None, None, :]
        x_1 = nn.silu(x_1)
        x_2 = nn.silu(self.gn2(self.cn2(x_1)))
        return x_2 + self.identity(x)


class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)

        # downsampling layers
        self.d_1_rb = Block(32, 32)
        self.d_1 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.d_2_rb = Block(64, 64)
        self.d_2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.d_3_rb = Block(128, 128)
        self.d_3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.d_4_rb = Block(256, 256)

        # bottleneck@256d
        self.bn = Block(256, 256)

        # upsampling layers
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.u_4_rb = Block(256 + 256, 128)
        self.u_3_rb = Block(128 + 128, 64)
        self.u_2_rb = Block(64 + 64, 32)

        self.gp = nn.GroupNorm(8, 32)
        self.out = nn.Conv2d(32, 1, kernel_size=1)

    def __call__(self, x: mx.array, time_embedding: mx.array):
        x = self.stem(x)
        skips = []
        for res, conv in [
            (self.d_1_rb, self.d_1),
            (self.d_2_rb, self.d_2),
            (self.d_3_rb, self.d_3),
            (self.d_4_rb, None),
        ]:
            x = res(x, time_embedding)
            skips.append(x)
            if conv is not None:
                x = conv(x)

        x = self.bn(x, time_embedding)
        for res in [self.u_4_rb, self.u_3_rb, self.u_2_rb]:
            skip = skips.pop()
            if x.shape[1] != skip.shape[1]:
                if x.shape[1] > skip.shape[1]:
                    x = x[:, : skip.shape[1], : skip.shape[2], :]

            x = mx.concat([x, skip], axis=-1)
            x = res(x, time_embedding)
            x = self.up(x)

        x = self.gp(x)
        x = self.out(nn.silu(x))
        return x


if __name__ == "__main__":
    T = 200
    B = 2
    x = mx.random.uniform(shape=(B, 28, 28, 1), dtype=mx.float32)
    t = mx.random.randint(0, T, shape=(B,), dtype=mx.int32)
    tembedder = TimeEmbedding(128, T)
    tembed = tembedder(t)
    network = UNET()
    out = network(x, tembed)
    print(out.shape)
