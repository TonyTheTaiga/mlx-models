import mlx.nn as nn
import numpy as np
import math


class ResidualEncoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.conv1 = nn.Conv2d(
            input_dim, output_dim, kernel_size=3, padding=1, bias=False
        )
        self.conv2 = nn.Conv2d(
            output_dim, output_dim, kernel_size=3, padding=1, bias=False
        )
        self.relu = nn.ReLU()
        self.p = nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def __call__(self, x):
        identity = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.relu(x + self.p(identity))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.conv = nn.Conv2d(
            input_dim, output_dim, kernel_size=3, padding=1, bias=False
        )
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
            layers.append(ResidualEncoderBlock(dims[idx], dims[idx + 1]))

        self.layers = layers

    def __call__(self, x):
        skip = []
        for layer in self.layers:
            x = layer(x)
            skip.append(x)

        return x, skip


class FSRCNNEncoder(nn.Module):
    def __init__(self, input_dims, output_dims):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(input_dims, output_dims, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(output_dims, output_dims // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(output_dims // 2, output_dims // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(output_dims // 2, output_dims // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(output_dims // 2, output_dims // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(output_dims // 2, output_dims // 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(output_dims // 2, output_dims, kernel_size=1),
            nn.ReLU(),
        )


class Decoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int):
        super().__init__()

        dims = np.geomspace(input_dim, 64, num_layers).tolist()
        dims = [math.ceil(dim) for dim in dims]  # pyright: ignore

        layers = []
        for idx in range(len(dims) - 1):
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(
                        dims[idx], dims[idx + 1], kernel_size=3, padding=1, stride=1
                    ),
                ]
            )

        layers.extend(
            [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(dims[-1], output_dim, kernel_size=3, padding=1, stride=1),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def __call__(self, x):
        return self.layers(x)


class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor: int):
        super().__init__()
        assert upscale_factor > 0

        self.upscale_factor = upscale_factor

    def __call__(self, x: "mx.array"):
        b, h, w, c = x.shape

        assert (
            c > self.upscale_factor**2
        ), f"feature dim must be greater than upscale_factor ^ 2"

        c_out = c // (self.upscale_factor**2)
        x = x.reshape(b, c_out, self.upscale_factor, self.upscale_factor, h, w)
        x = x.transpose(0, 1, 4, 2, 5, 3)
        x = x.reshape(b, c_out, h * self.upscale_factor, w * self.upscale_factor)
        return x


def pixel_shuffle(x: "mx.array", upscale_factor: int):
    b, h, w, c = x.shape

    assert c > upscale_factor**2, f"feature dim must be greater than upscale_factor ^ 2"

    c_out = c // (upscale_factor**2)
    x = x.reshape(b, c_out, upscale_factor, upscale_factor, h, w)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    x = x.reshape(b, c_out, h * upscale_factor, w * upscale_factor)
    return x


class SuperResolution(nn.Module):
    def __init__(
        self,
        upscale: int,
        num_encoder_layers: int = 3,
        latent_dim: int = 256,
        bottleneck: bool = False,
    ):
        super().__init__()

        self.encoder = Encoder(3, latent_dim, num_encoder_layers)

        if bottleneck:
            self.bottleneck = nn.Sequential(
                nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(),
                nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1, bias=False),
                nn.ReLU(),
            )
        else:
            self.bottleneck = nn.Identity()

        self.decoder = Decoder(latent_dim, 3, upscale)

    def __call__(self, x):
        x, _ = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x


# if __name__ == "__main__":
#     import mlx.core as mx

#     model = SuperResolution(upscale=2)
#     _input = mx.zeros(shape=(1, 256, 160, 3))
#     output = model(_input)

#     _input = mx.zeros(shape=(1, 256, 160, 3 * (2**2)))
#     print(pixel_shuffle(_input, 2).shape)
