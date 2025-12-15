import mlx.core as mx
import mlx.nn as nn

from networks.speech_autoencoder.cnext import CausalConvNeXtBlock, ConvNeXtBlock


class Encoder(nn.Module):
    def __init__(self, in_dims: int, out_dims: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=in_dims, out_channels=512, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm(num_features=512)
        self.convnext_blocks = nn.Sequential(
            *[ConvNeXtBlock(dim=512, expansion=4) for _ in range(10)]
        )
        self.project = nn.Linear(input_dims=512, output_dims=out_dims)
        self.ln1 = nn.LayerNorm(dims=out_dims)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.convnext_blocks(x)
        x = self.project(x)
        x = self.ln1(x)

        return x


class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def __call__(self, x: mx.array) -> mx.array:
        pad_width = [(0, 0), (self.left_pad, 0), (0, 0)]
        x = mx.pad(x, pad_width=pad_width, constant_values=0)
        return self.conv(x)


class Decoder(nn.Module):
    def __init__(self, in_dims: int, out_dims: int):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels=in_dims, out_channels=512, kernel_size=7)
        self.bn1 = nn.BatchNorm(512)
        self.convnext_blocks = nn.Sequential(
            *[
                CausalConvNeXtBlock(dim=512, expansion=4, kernel_size=7, dilation=dilation)
                for dilation in [1, 2, 4, 1, 2, 4, 1, 1, 1, 1]
            ]
        )
        self.bn2 = nn.BatchNorm(512)
        self.conv2 = CausalConv1d(in_channels=512, out_channels=2048, kernel_size=3)
        self.act = nn.PReLU()
        self.linear = nn.Linear(input_dims=2048, output_dims=out_dims)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.convnext_blocks(x)
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.act(x)
        x = self.linear(x)
        bsz, seq_len, feat = x.shape
        waveform = mx.reshape(x, (bsz, seq_len * feat, 1))
        waveform = mx.tanh(waveform)
        return waveform


class SpeechAutoEncoder(nn.Module):
    def __init__(self, in_dims: int, hidden_dims: int = 24, out_dims: int = 512):
        super().__init__()
        self.encoder = Encoder(in_dims=in_dims, out_dims=hidden_dims)
        self.decoder = Decoder(in_dims=hidden_dims, out_dims=out_dims)

    def __call__(self, x: mx.array):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == "__main__":
    batch_size = 2
    sequence_length = 400
    mel_bins = 228

    ae = SpeechAutoEncoder(in_dims=mel_bins)
    x = mx.random.uniform(
        low=0.0,
        high=1.0,
        shape=(batch_size, 400, 228),
    )
    print(f"Input Shape: {x.shape}")
    output = ae(x)
    print(f"AE output shape: {output.shape}")
