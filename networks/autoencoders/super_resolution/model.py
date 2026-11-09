import math

import mlx.core as mx
import mlx.nn as nn


class ResidualEncoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim, res_scale: float = 0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.res_scale = res_scale

    def __call__(self, x):
        shortcut = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = shortcut + (x * self.res_scale)
        return x


class Encoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int):
        super().__init__()

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)  # stem

        layers = []
        for _ in range(num_layers):
            layers.append(ResidualEncoderBlock(output_dim, output_dim))

        self.layers = nn.Sequential(*layers)
        self.conv_out = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)

    def __call__(self, x):
        x = self.conv(x)
        shortcut = x
        x = self.layers(x)
        x = self.conv_out(x)
        return x + shortcut


class Decoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, scale_factor: int):
        super().__init__()

        num_upsamples = int(math.log2(scale_factor))
        if 2**num_upsamples != scale_factor:
            raise ValueError("scale_factor must be a power of two")

        # nearest + conv instead of pixel shuffle, which gave checkerboard artifacts under GAN loss
        layers = []
        for _ in range(num_upsamples):
            layers.extend(
                [
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(input_dim, input_dim, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2),
                ]
            )

        self.layers = nn.Sequential(*layers)
        self.final = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)

    def __call__(self, x):
        x = self.layers(x)
        return self.final(x)


class SuperResolution(nn.Module):
    def __init__(
        self,
        upscale: int,
        num_encoder_layers: int = 3,
        latent_dim: int = 256,
    ):
        super().__init__()

        self.encoder = Encoder(3, latent_dim, num_encoder_layers)
        self.decoder = Decoder(latent_dim, 3, upscale)
        # zero init: the residual starts at 0, so the output starts as plain bicubic
        self.decoder.final.weight = mx.zeros_like(self.decoder.final.weight)
        self.decoder.final.bias = mx.zeros_like(self.decoder.final.bias)
        self.upsample = nn.Upsample(scale_factor=upscale, mode="cubic", align_corners=False)

    def __call__(self, x):
        residual = self.decoder(self.encoder(x))
        return self.upsample(x) + residual


if __name__ == "__main__":
    import mlx.core as mx

    model = SuperResolution(upscale=4)
    _input = mx.zeros(shape=(1, 256, 160, 3))
    output = model(_input)
