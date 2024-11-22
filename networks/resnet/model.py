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
        self.relu = nn.ReLU()

    def __call__(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, shortcut):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm(out_channels // 4)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels // 4,
            out_channels=out_channels // 4,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm(out_channels // 4)
        self.conv3 = nn.Conv2d(
            in_channels=out_channels // 4,
            out_channels=out_channels,
            stride=1,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm(out_channels)

        self.shortcut = shortcut
        self.relu = nn.ReLU()

    def __call__(self, x):
        identity = self.shortcut(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = x + identity
        return self.relu(x)


class Shortcut(nn.Module):
    def __init__(self, n_dim_padding: int):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=1, stride=2, padding=0)
        self.n_dim_padding = n_dim_padding

    def __call__(self, x: mx.array):
        x = self.pool(x)
        x = mx.pad(x, pad_width=[(0, 0), (0, 0), (0, 0), (0, self.n_dim_padding)])
        return x


class ConvShortcut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.bn = nn.BatchNorm(out_channels)

    def __call__(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return self.bn(x)


class Layer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: int,
        num_blocks: int,
        block: nn.Module,
    ):
        super().__init__()

        blocks = []
        blocks.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=2 if downsample else 1,
                shortcut=ConvShortcut(in_channels, out_channels)
                # shortcut=Shortcut(out_channels-in_channels)
                if downsample
                else nn.Identity(),
            )
        )

        for _ in range(num_blocks - 1):
            blocks.append(
                block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    shortcut=nn.Identity(),
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

        # stem as proposed in resnet-c
        self.stem = nn.Sequential(
            *[
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm(32),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm(32),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=init_layer_outsize,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm(init_layer_outsize),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            ]
        )

        layer1_outsize = l1_size
        self.layer1 = Layer(
            in_channels=init_layer_outsize,
            out_channels=layer1_outsize,
            downsample=False,
            num_blocks=3,
            block=Block,
        )

        layer2_outsize = l2_size
        self.layer2 = Layer(
            in_channels=layer1_outsize,
            out_channels=layer2_outsize,
            downsample=True,
            num_blocks=4,
            block=Block,
        )

        layer3_outsize = l3_size
        self.layer3 = Layer(
            in_channels=layer2_outsize,
            out_channels=layer3_outsize,
            downsample=True,
            num_blocks=6,
            block=Block,
        )

        layer4_outsize = l4_size
        self.layer4 = Layer(
            in_channels=layer3_outsize,
            out_channels=layer4_outsize,
            downsample=True,
            num_blocks=3,
            block=Block,
        )

        self.classifier = nn.Linear(layer4_outsize, num_classes)

    def __call__(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = mx.mean(x, (1, 2))
        x = self.classifier(x)
        return x


class ResNet50(nn.Module):
    def __init__(self, in_channels, num_classes, layer_sizes=[256, 512, 1024, 2048]):
        super().__init__()
        l1_size, l2_size, l3_size, l4_size = layer_sizes

        init_layer_outsize = l1_size

        # stem as proposed in resnet-c
        self.init_layer = nn.Sequential(
            *[
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=32,
                    kernel_size=3,
                    stride=2,
                    padding=3,
                    bias=False,
                ),
                nn.BatchNorm(32),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm(32),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=init_layer_outsize,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm(init_layer_outsize),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
            ]
        )

        layer1_outsize = l1_size
        self.layer1 = Layer(
            in_channels=init_layer_outsize,
            out_channels=layer1_outsize,
            downsample=False,
            num_blocks=3,
            block=BottleneckBlock,
        )

        layer2_outsize = l2_size
        self.layer2 = Layer(
            in_channels=layer1_outsize,
            out_channels=layer2_outsize,
            downsample=True,
            num_blocks=4,
            block=BottleneckBlock,
        )

        layer3_outsize = l3_size
        self.layer3 = Layer(
            in_channels=layer2_outsize,
            out_channels=layer3_outsize,
            downsample=True,
            num_blocks=6,
            block=BottleneckBlock,
        )

        layer4_outsize = l4_size
        self.layer4 = Layer(
            in_channels=layer3_outsize,
            out_channels=layer4_outsize,
            downsample=True,
            num_blocks=3,
            block=BottleneckBlock,
        )

        self.classifier = nn.Linear(layer4_outsize, num_classes)

    def __call__(self, x):
        x = self.init_layer(x)
        print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = mx.mean(x, (1, 2))
        x = self.classifier(x)
        return x
