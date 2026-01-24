import mlx.core as mx
import mlx.nn as nn

from networks.speech_autoencoder.cnext import CausalConvNeXtBlock, ConvNeXtBlock
from training.speech_autoencoder.utils import hann_window


class Encoder(nn.Module):
    def __init__(self, in_dims: int, out_dims: int) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=in_dims, out_channels=512, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm(512)
        self.convnext_blocks = nn.Sequential(
            *[ConvNeXtBlock(dim=512, expansion=4) for _ in range(8)]
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


class OverlapAddFlatteningHead(nn.Module):
    """
    Predicts overlapping waveform frames and overlap-adds them into a continuous waveform.

    Input:  x  shape (B, T_frames, C)
    Output: y  shape (B, T_samples, 1)

    You MUST set hop_length to match the mel hop (in samples) used for the mel you feed the AE.
    """

    def __init__(
        self,
        in_channels: int,
        frame_length: int,
        hop_length: int,
        out_channels: int = 1,
        window: str = "hann",
        normalize: bool = True,
    ):
        super().__init__()
        if frame_length <= 0 or hop_length <= 0:
            raise ValueError("frame_length and hop_length must be > 0")
        if out_channels != 1:
            raise ValueError("This head is written for mono (out_channels=1).")

        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.normalize = bool(normalize)

        # Map features -> samples in a frame (frame_length)
        self.to_frame = nn.Linear(input_dims=in_channels, output_dims=self.frame_length)

        if window == "hann":
            w = hann_window(self.frame_length, dtype=mx.float32)  # (frame_length,)
        elif window == "rect":
            w = mx.ones((self.frame_length,), dtype=mx.float32)
        else:
            raise ValueError(f"Unknown window: {window}")

        self.window = w  # mx.array

    def __call__(self, x: mx.array) -> mx.array:
        B, T, C = x.shape

        frames = self.to_frame(x)  # (B, T, frame_length)
        frames = frames * self.window[None, None, :]  # windowing

        # Output length after overlap-add
        out_len = (T - 1) * self.hop_length + self.frame_length
        y = mx.zeros((B, out_len), dtype=frames.dtype)

        # Overlap-add (simple loop; T is usually a few hundred, acceptable)
        for t in range(T):
            start = t * self.hop_length
            y[:, start : start + self.frame_length] = (
                y[:, start : start + self.frame_length] + frames[:, t, :]
            )

        if self.normalize:
            # Compute overlap gain to normalize amplitude (window squared sum)
            denom = mx.zeros((out_len,), dtype=frames.dtype)
            w2 = (self.window**2).astype(frames.dtype)
            for t in range(T):
                start = t * self.hop_length
                denom[start : start + self.frame_length] = (
                    denom[start : start + self.frame_length] + w2
                )

            denom = mx.maximum(denom, mx.array(1e-6, dtype=denom.dtype))
            y = y / denom[None, :]

        return y[:, :, None]  # (B, out_len, 1)


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
