import mlx.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int):
        super().__init__()

    def __call__(self, x): ...


class Decoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int):
        super().__init__()

    def __call__(self, x): ...


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
        x = self.encoder(x)
        x = self.decoder(x)
        return x
