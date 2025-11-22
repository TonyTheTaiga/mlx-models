from collections.abc import Callable
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class Block(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        expansion_factor: int,
        stride: int,
        padding: int,
        nonlinearlity: Callable,
        se: bool,
    ):
        super().__init__()

        # se = squeeze and excite
        assert stride == 1 or stride == 2, "only stride of 1 or 2 is supported"

        self.stride = stride
        self.in_dim = in_dim
        self.out_dim = out_dim

        layers = []
        if expansion_factor != 1:
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels=in_dim,
                        out_channels=in_dim * expansion_factor,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False,
                    ),
                    nn.BatchNorm(in_dim * expansion_factor),
                    nonlinearlity,
                ]
            )

        layers.extend(
            [
                nn.Conv2d(
                    in_channels=in_dim * expansion_factor,
                    out_channels=in_dim * expansion_factor,
                    kernel_size=3,
                    stride=stride,
                    padding=padding,
                    bias=False,
                    groups=in_dim * expansion_factor,
                ),
                nn.BatchNorm(in_dim * expansion_factor),
                nonlinearlity,
            ]
        )

        layers.extend(
            [
                nn.Conv2d(
                    in_channels=in_dim * expansion_factor,
                    out_channels=out_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                ),
                nn.BatchNorm(out_dim),
            ]
        )

        self.layers = nn.Sequential(*layers)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.layers(x)
        if self.stride == 1 and self.in_dim == self.out_dim:
            return x + y

        return y


@dataclass(kw_only=True)
class MobileNetConfig:
    expansion_factor: int
    channels: int
    repeated: int
    stride: int


PAPER_CONFIG = [
    MobileNetConfig(expansion_factor=1, channels=16, repeated=1, stride=1),
    MobileNetConfig(expansion_factor=6, channels=24, repeated=2, stride=2),
    MobileNetConfig(expansion_factor=6, channels=32, repeated=2, stride=2),
    MobileNetConfig(expansion_factor=6, channels=64, repeated=4, stride=2),
    MobileNetConfig(expansion_factor=6, channels=96, repeated=2, stride=1),
    MobileNetConfig(expansion_factor=6, channels=160, repeated=2, stride=2),
    MobileNetConfig(expansion_factor=6, channels=320, repeated=1, stride=1),
]


class MobileNet(nn.Module):
    def __init__(self, num_classes: int, config: list[MobileNetConfig]):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm(32),
            nn.ReLU6(),
        )

        bottlenecks = []
        in_dim = 32
        for stage in config:
            for repeat_idx in range(stage.repeated):
                stride = stage.stride if repeat_idx == 0 else 1
                bottlenecks.append(
                    Block(
                        in_dim=in_dim,
                        out_dim=stage.channels,
                        expansion_factor=stage.expansion_factor,
                        stride=stride,
                        padding=1,
                        nonlinearlity=nn.ReLU6(),
                        se=False,
                    )
                )
                in_dim = stage.channels

        self.bottlenecks = nn.Sequential(*bottlenecks)

        self.output = nn.Sequential(
            nn.Conv2d(
                in_channels=config[-1].channels,
                out_channels=1280,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm(1280),
            nn.ReLU6(),
        )
        self.classifier = nn.Linear(1280, num_classes)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.stem(x)
        y = self.bottlenecks(y)
        y = self.output(y)
        y = mx.mean(y, (1, 2))
        y = self.classifier(y)
        return y


if __name__ == "__main__":
    block = MobileNet(num_classes=10, config=PAPER_CONFIG)
    _input = mx.random.normal(shape=(4, 224, 224, 3))
    print(block(_input).shape)
    num_params = sum(v.size for _, v in tree_flatten(block.parameters()))
    print(f"num params = {num_params}")
