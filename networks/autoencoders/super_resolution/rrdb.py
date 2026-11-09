import math

import mlx.core as mx
import mlx.nn as nn


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_features: int = 64, growth: int = 32, res_scale: float = 0.2):
        super().__init__()

        self.convs = [
            nn.Conv2d(num_features + i * growth, growth, kernel_size=3, padding=1) for i in range(4)
        ]
        self.conv_out = nn.Conv2d(num_features + 4 * growth, num_features, kernel_size=3, padding=1)
        self.res_scale = res_scale

    def __call__(self, x):
        features = [x]
        for conv in self.convs:
            features.append(nn.leaky_relu(conv(mx.concatenate(features, axis=-1)), 0.2))
        return x + self.res_scale * self.conv_out(mx.concatenate(features, axis=-1))


class RRDB(nn.Module):
    def __init__(self, num_features: int, growth: int, res_scale: float = 0.2):
        super().__init__()

        self.blocks = nn.Sequential(
            *[ResidualDenseBlock(num_features, growth, res_scale) for _ in range(3)]
        )
        self.res_scale = res_scale

    def __call__(self, x):
        return x + self.res_scale * self.blocks(x)


class RRDBNet(nn.Module):
    def __init__(
        self,
        upscale: int,
        num_blocks: int = 23,
        num_features: int = 64,
        growth: int = 32,
    ):
        super().__init__()

        num_upsamples = int(math.log2(upscale))
        if 2**num_upsamples != upscale:
            raise ValueError("upscale must be a power of two")

        self.conv_first = nn.Conv2d(3, num_features, kernel_size=3, padding=1)
        self.trunk = nn.Sequential(*[RRDB(num_features, growth) for _ in range(num_blocks)])
        self.conv_trunk = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        self.upsamples = [
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
            for _ in range(num_upsamples)
        ]
        self.conv_hr = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.conv_last = nn.Conv2d(num_features, 3, kernel_size=3, padding=1)
        # zero init: the residual starts at 0, so the output starts as plain bicubic
        self.conv_last.weight = mx.zeros_like(self.conv_last.weight)
        self.conv_last.bias = mx.zeros_like(self.conv_last.bias)

        self.nearest = nn.Upsample(scale_factor=2, mode="nearest")
        self.bicubic = nn.Upsample(scale_factor=upscale, mode="cubic", align_corners=False)

    def __call__(self, x):
        features = self.conv_first(x)
        features = features + self.conv_trunk(self.trunk(features))
        for conv in self.upsamples:
            features = nn.leaky_relu(conv(self.nearest(features)), 0.2)
        residual = self.conv_last(nn.leaky_relu(self.conv_hr(features), 0.2))
        return self.bicubic(x) + residual
