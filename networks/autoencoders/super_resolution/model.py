import mlx.nn as nn
import mlx.core as mx
import math


class ResidualEncoderBlock(nn.Module):
    def __init__(self, input_dim, output_dim, res_scale: float=0.1):
        super().__init__()

        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.res_scale = res_scale

    def __call__(self, x):
        identity = x
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = identity + (x * self.res_scale)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int):
        super().__init__()

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)
        layers = []
        for _ in range(num_layers):
            layers.append(ResidualEncoderBlock(output_dim, output_dim))

        self.layers = layers

    def __call__(self, x):
        skip = []
        x = self.conv(x)
        identity = x
        for layer in self.layers:
            x = layer(x)
            skip.append(x)

        return x + identity, identity, skip


class Decoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, scale_factor: int):
        super().__init__()

        num_upsamples = int(math.log2(scale_factor))
        layers = []
        for _ in range(num_upsamples):
            layers.extend(
                [
                    nn.Conv2d(input_dim, input_dim * 4, kernel_size=3, padding=1),
                    PixelShuffle(2),
                ]
            )
        # self.upscale = nn.Upsample(scale_factor=scale_factor, mode='cubic')
        self.final = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)
        # layers.extend([nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1, stride=1)])
        self.layers = nn.Sequential(*layers)



    def __call__(self, x):
        x = self.layers(x)
        return self.final(x)


class PixelShuffle(nn.Module):
    def __init__(self, upscale_factor: int):
        super().__init__()
        assert upscale_factor > 0

        self.upscale_factor = upscale_factor

    def __call__(self, x: mx.array):
        return pixel_shuffle(x, self.upscale_factor)


def pixel_shuffle(x: mx.array, upscale_factor: int):
    b, h, w, c = x.shape
    assert c % (upscale_factor**2) == 0, f"feature dim must be divisible by upscale_factor ^ 2"
    c_out = c // (upscale_factor**2)
    x = x.reshape(b, h, w, c_out, upscale_factor, upscale_factor)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    x = x.reshape(b, h * upscale_factor, w * upscale_factor, c_out)
    return x


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

    def __call__(self, x):
        x, _, _ = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    import mlx.core as mx
    from mlx.utils import tree_flatten

    model = SuperResolution(upscale=4)
    _input = mx.zeros(shape=(1, 256, 160, 3))
    output = model(_input)
    for k, v in tree_flatten(model.parameters()):
        print(k, v.shape)

#     _input = mx.zeros(shape=(1, 256, 160, 3 * (2**2)))
#     print(pixel_shuffle(_input, 2).shape)
