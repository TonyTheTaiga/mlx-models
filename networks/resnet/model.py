import mlx.core as mx
import mlx.nn as nn


class Block(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride: int, shortcut: nn.Module | None
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm(out_channels)
        self.shortcut = shortcut

    def __call__(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        if self.shortcut is not None:
            residual = self.shortcut(residual)

        x += residual

        x = self.relu(x)
        return x


# TODO: Implement BottleneckBlock as found in larger resnets
class BottleneckBlock(nn.Module):
    pass


class Shortcut(nn.Module):
    def __init__(self, n_dim_padding: int):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=1, stride=2, padding=0)
        self.n_dim_padding = n_dim_padding

    def __call__(self, x):
        x = self.pool(x)
        # projecting might be less dynamic
        b, h, w, _ = x.shape
        padding = mx.zeros((b, h, w, self.n_dim_padding), dtype=mx.float32)
        x = mx.concatenate([x, padding], axis=-1)

        return x


class ConvShortcut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=2,
            bias=False,
        )
        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x):
        x = self.conv(x)
        return self.bn(x)


class Layer(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, downsample: int, num_blocks: int
    ):
        super().__init__()

        blocks = []
        blocks.append(
            Block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2 if downsample else 1,
                shortcut=ConvShortcut(in_channels, out_channels)
                if in_channels != out_channels
                else None,
            )
        )

        for _ in range(num_blocks - 1):
            blocks.append(
                Block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    shortcut=None,
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def __call__(self, x):
        return self.blocks(x)


class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, layer_sizes=[64, 128, 256, 512]):
        super().__init__()
        l1_size, l2_size, l3_size, l4_size = layer_sizes

        init_layer_outsize = l1_size
        self.init_layer = nn.Sequential(
            *[
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=init_layer_outsize,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                ),
                nn.BatchNorm(init_layer_outsize),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )

        layer1_outsize = l1_size
        self.layer1 = Layer(
            in_channels=init_layer_outsize,
            out_channels=layer1_outsize,
            downsample=False,
            num_blocks=3,
        )

        layer2_outsize = l2_size
        self.layer2 = Layer(
            in_channels=layer1_outsize,
            out_channels=layer2_outsize,
            downsample=True,
            num_blocks=4,
        )

        layer3_outsize = l3_size
        self.layer3 = Layer(
            in_channels=layer2_outsize,
            out_channels=layer3_outsize,
            downsample=True,
            num_blocks=6,
        )

        layer4_outsize = l4_size
        self.layer4 = Layer(
            in_channels=layer3_outsize,
            out_channels=layer4_outsize,
            downsample=True,
            num_blocks=3,
        )

        self.classifier = nn.Linear(layer4_outsize, num_classes)

    def __call__(self, x):
        x = self.init_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = mx.mean(x, (1, 2))
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    network = ResNet(3, num_classes=10)
    t = mx.random.uniform(0.0, 1.0, shape=(4, 224, 224, 3))
    logits = network(t)
    print(logits.shape)
