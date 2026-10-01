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
