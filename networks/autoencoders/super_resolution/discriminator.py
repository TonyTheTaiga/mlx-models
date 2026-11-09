import mlx.core as mx
import mlx.nn as nn


class Discriminator(nn.Module):
    """
    Fully convolutional patch discriminator in the style of SRGAN: pairs of 3x3 convs, the second
    of each pair strided, widening as resolution drops. Outputs one realness score per patch, so it
    works on any crop size.
    """

    def __init__(self, base_dim: int = 64, num_stages: int = 4):
        super().__init__()

        layers = [nn.Conv2d(3, base_dim, kernel_size=3, padding=1), nn.LeakyReLU(0.2)]
        in_dim = base_dim
        for stage in range(num_stages):
            out_dim = base_dim * min(2**stage, 8)
            layers.extend(
                [
                    nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(out_dim, out_dim, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2),
                ]
            )
            in_dim = out_dim

        self.layers = nn.Sequential(*layers)
        self.head = nn.Conv2d(in_dim, 1, kernel_size=3, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        return self.head(self.layers(x))


def _normalize(x: mx.array, eps: float = 1e-12) -> mx.array:
    return x / (mx.linalg.norm(x) + eps)


class SpectralNormConv2d(nn.Module):
    """
    Conv2d whose weight is divided by its largest singular value (Miyato et al., 2018), which caps
    how sharply the discriminator can react and keeps it from overpowering the generator.

    The singular vectors u, v are frozen state, refreshed by `power_iteration()` once per training
    step outside of gradient transforms; the forward pass treats them as constants.
    """

    def __init__(
        self, in_dim: int, out_dim: int, kernel_size: int, stride: int = 1, padding: int = 0
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding)
        self.stride = stride
        self.padding = padding
        self.u = _normalize(mx.random.normal((out_dim,)))
        self.v = _normalize(mx.random.normal((in_dim * kernel_size * kernel_size,)))
        self.freeze(keys=["u", "v"], recurse=False)
        for _ in range(10):  # converge to the top singular vectors before the first step
            self.power_iteration()

    def _matrix(self) -> mx.array:
        return self.conv.weight.reshape(self.conv.weight.shape[0], -1)

    def power_iteration(self) -> None:
        w = self._matrix()
        self.v = _normalize(w.T @ self.u)
        self.u = _normalize(w @ self.v)

    def __call__(self, x: mx.array) -> mx.array:
        sigma = self.u @ (self._matrix() @ self.v)
        weight = self.conv.weight / sigma
        y = mx.conv2d(x, weight, stride=self.stride, padding=self.padding)
        return y + self.conv.bias


class UNetDiscriminator(nn.Module):
    """
    U-Net discriminator with spectral norm, from Real-ESRGAN (Wang et al., 2021,
    https://arxiv.org/abs/2107.10833). It outputs a realness score per pixel instead of per patch,
    so the generator gets local feedback on exactly where its texture looks fake.
    """

    def __init__(self, base_dim: int = 64):
        super().__init__()

        d = base_dim
        self.conv0 = nn.Conv2d(3, d, kernel_size=3, padding=1)
        # Downsampling path.
        self.conv1 = SpectralNormConv2d(d, d * 2, 4, stride=2, padding=1)
        self.conv2 = SpectralNormConv2d(d * 2, d * 4, 4, stride=2, padding=1)
        self.conv3 = SpectralNormConv2d(d * 4, d * 8, 4, stride=2, padding=1)
        # Upsampling path with skip connections.
        self.conv4 = SpectralNormConv2d(d * 8, d * 4, 3, padding=1)
        self.conv5 = SpectralNormConv2d(d * 4, d * 2, 3, padding=1)
        self.conv6 = SpectralNormConv2d(d * 2, d, 3, padding=1)
        self.conv7 = SpectralNormConv2d(d, d, 3, padding=1)
        self.conv8 = SpectralNormConv2d(d, d, 3, padding=1)
        self.conv9 = nn.Conv2d(d, 1, kernel_size=3, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)

    def power_iteration(self) -> None:
        for _, module in self.named_modules():
            if isinstance(module, SpectralNormConv2d):
                module.power_iteration()

    def __call__(self, x: mx.array) -> mx.array:
        act = lambda t: nn.leaky_relu(t, 0.2)  # noqa: E731
        x0 = act(self.conv0(x))
        x1 = act(self.conv1(x0))
        x2 = act(self.conv2(x1))
        x3 = act(self.conv3(x2))

        x4 = act(self.conv4(self.up(x3))) + x2
        x5 = act(self.conv5(self.up(x4))) + x1
        x6 = act(self.conv6(self.up(x5))) + x0

        out = act(self.conv7(x6))
        out = act(self.conv8(out))
        return self.conv9(out)
