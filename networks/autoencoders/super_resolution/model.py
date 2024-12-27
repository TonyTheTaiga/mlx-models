import mlx.nn as nn
import numpy as np
import math


class ResidualEncoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.project = nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def __call__(self, x):
        identity = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.relu(self.project(identity) + x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def __call__(self, x):
        return self.relu(self.conv(x))


class Encoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int):
        super().__init__()

        dims = np.geomspace(64, output_dim, num_layers)
        dims = [math.ceil(dim) for dim in dims]  # pyright: ignore

        layers = []
        layers.append(EncoderBlock(input_dim, 64))
        for idx in range(len(dims) - 1):
            layers.append(EncoderBlock(dims[idx], dims[idx + 1]))

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
            layers.extend([nn.ConvTranspose2d(dims[idx], dims[idx + 1], kernel_size=4, stride=2, padding=1), nn.ReLU()])

        layers.extend([nn.ConvTranspose2d(dims[-1], output_dim, kernel_size=4, stride=2, padding=1), nn.ReLU()])
        self.layers = nn.Sequential(*layers)

    def __call__(self, x):
        return self.layers(x)


class SuperResolution(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        output_dim: int,
        latent_dim: int,
    ):
        super().__init__()

        self.encoder = Encoder(input_dim, latent_dim, num_encoder_layers)
        self.decoder = Decoder(latent_dim, output_dim, num_decoder_layers)

    def __call__(self, x):
        x, _ = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    import mlx.core as mx

    model = SuperResolution(input_dim=3, num_encoder_layers=3, num_decoder_layers=3, output_dim=3, latent_dim=256)
    _input = mx.zeros(shape=(1, 256, 160, 3))
    output = model(_input)
