import mlx.nn as nn
import numpy as np
import math


class ResidualEncoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm(output_dim)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm(output_dim)
        self.relu = nn.ReLU()
        self.p = nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def __call__(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.relu(x + self.p(identity))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm(output_dim)
        self.relu = nn.ReLU()

    def __call__(self, x):
        return self.relu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int):
        super().__init__()

        dims = np.geomspace(64, output_dim, num_layers)
        dims = [math.ceil(dim) for dim in dims]  # pyright: ignore

        layers = []
        layers.append(EncoderBlock(input_dim, 64))
        for idx in range(len(dims) - 1):
            layers.append(ResidualEncoderBlock(dims[idx], dims[idx + 1]))

        self.layers = layers

    def __call__(self, x):
        skip = []
        for layer in self.layers:
            x = layer(x)
            skip.append(x)

        return x, skip


class Decoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int):
        super().__init__()

        dims = np.geomspace(input_dim, 64, num_layers).tolist()
        dims = [math.ceil(dim) for dim in dims]  # pyright: ignore

        layers = []
        for idx in range(len(dims) - 1):
            layers.extend(
                [
                    nn.ConvTranspose2d(dims[idx], dims[idx + 1], kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm(dims[idx + 1]),
                    nn.ReLU(),
                ]
            )

        layers.extend(
            [
                nn.ConvTranspose2d(dims[-1], output_dim, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def __call__(self, x):
        return self.layers(x)


class SuperResolution(nn.Module):
    def __init__(self, upscale: int, num_encoder_layers: int = 3, latent_dim: int = 256):
        super().__init__()

        self.encoder = Encoder(3, latent_dim, num_encoder_layers)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm(latent_dim // 2),
            nn.ReLU(),
            nn.Conv2d(latent_dim // 2, latent_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm(latent_dim),
            nn.ReLU(),
        )
        self.decoder = Decoder(latent_dim, 3, upscale)

    def __call__(self, x):
        x, _ = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    import mlx.core as mx

    model = SuperResolution(upscale=2)
    _input = mx.zeros(shape=(1, 256, 160, 3))
    output = model(_input)
    print(output.shape)
