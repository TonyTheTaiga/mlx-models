import mlx.core as mx
import mlx.nn as nn

from training.speech_autoencoder.utils import ensure_waveform_2d, hann_window, reflect_pad_1d


class MRD(nn.Module):
    class Helper(nn.Module):
        def __init__(self, fft_size):
            super().__init__()
            self.fft_size = fft_size
            self.hop_size = int(fft_size * 0.25)
            self.window_length = fft_size
            self._window = hann_window(self.window_length, dtype=mx.float32)
            self.layers = [
                nn.Conv2d(
                    in_channels=1,
                    out_channels=16,
                    kernel_size=(5, 5),
                    stride=(1, 1),
                    padding=(2, 2),
                ),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=(5, 5),
                    stride=(2, 1),
                    padding=(2, 2),
                ),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=(5, 5),
                    stride=(2, 1),
                    padding=(2, 2),
                ),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=(5, 5),
                    stride=(2, 1),
                    padding=(2, 2),
                ),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=16,
                    out_channels=16,
                    kernel_size=(5, 5),
                    stride=(1, 1),
                    padding=(2, 2),
                ),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=16,
                    out_channels=1,
                    kernel_size=(3, 3),
                    stride=(1, 1),
                    padding=(1, 1),
                ),
            ]

        @staticmethod
        def _frame_signal(x: mx.array, frame_length: int, hop_length: int) -> mx.array:
            if frame_length <= 0 or hop_length <= 0:
                raise ValueError("frame_length and hop_length must be positive")

            t = x.shape[1]
            if t < frame_length:
                pad = frame_length - t
            else:
                remainder = (t - frame_length) % hop_length
                pad = 0 if remainder == 0 else hop_length - remainder
            if pad:
                x = mx.pad(x, pad_width=[(0, 0), (0, pad)], constant_values=0)

            num_frames = 1 + (x.shape[1] - frame_length) // hop_length
            base = mx.arange(num_frames, dtype=mx.int32)[:, None] * int(hop_length)
            offsets = mx.arange(frame_length, dtype=mx.int32)[None, :]
            indices = base + offsets
            flat = indices.reshape((-1,))
            gathered = mx.take(x, flat, axis=1)
            return gathered.reshape((x.shape[0], num_frames, frame_length))

        def log_linear_spectrogram(self, waveform: mx.array) -> mx.array:
            pad = max((self.window_length - self.hop_size) // 2, 0)
            if pad:
                waveform = reflect_pad_1d(waveform, pad, pad, axis=1)
            frames = self._frame_signal(
                waveform, frame_length=self.window_length, hop_length=self.hop_size
            )
            frames = frames * self._window[None, None, :]
            spec = mx.fft.rfft(frames, n=self.fft_size, axis=-1)
            mag = mx.abs(spec)
            log_mag = mx.log(mag + mx.array(1e-7, dtype=mag.dtype))
            log_mag = mx.transpose(log_mag, (0, 2, 1))
            return log_mag[..., None]

        def __call__(self, x: mx.array) -> tuple[mx.array, list[mx.array]]:
            features: list[mx.array] = []
            for layer in self.layers:
                x = layer(x)
                if isinstance(layer, nn.Conv2d):
                    features.append(x)
            logits = x
            return logits, features

    def __init__(self, fft_sizes: list[int] = [512, 1024, 2048]):
        super().__init__()
        self.helpers = [self.Helper(fft_size=fft_size) for fft_size in fft_sizes]

    def __call__(self, waveform: mx.array) -> list[dict[str, object]]:
        waveform = ensure_waveform_2d(waveform)
        outputs: list[dict[str, object]] = []

        for helper in self.helpers:
            spectrogram = helper.log_linear_spectrogram(waveform)
            logits, features = helper(spectrogram)
            outputs.append(
                {
                    "fft_size": helper.fft_size,
                    "spectrogram": spectrogram,
                    "logits": logits,
                    "features": features,
                }
            )

        return outputs


class MPD(nn.Module):
    class Helper(nn.Module):
        def __init__(self, period: int):
            super().__init__()
            if period <= 0:
                raise ValueError("period must be positive")
            self.period = int(period)

            k = (5, 1)
            p = (2, 0)
            self.layers = [
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=k, stride=(3, 1), padding=p),
                nn.LeakyReLU(),
                nn.Conv2d(in_channels=16, out_channels=64, kernel_size=k, stride=(3, 1), padding=p),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=64, out_channels=256, kernel_size=k, stride=(3, 1), padding=p
                ),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=256, out_channels=512, kernel_size=k, stride=(3, 1), padding=p
                ),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=512, out_channels=512, kernel_size=k, stride=(1, 1), padding=p
                ),
                nn.LeakyReLU(),
                nn.Conv2d(
                    in_channels=512,
                    out_channels=1,
                    kernel_size=(3, 1),
                    stride=(1, 1),
                    padding=(1, 0),
                ),
            ]

        def _reshape_period(self, waveform: mx.array) -> mx.array:
            if waveform.ndim != 2:
                raise ValueError("MPD helper expects waveform shaped (B, T)")
            bsz, t = waveform.shape
            p = self.period
            remainder = t % p
            if remainder:
                pad = p - remainder
                waveform = mx.pad(waveform, pad_width=[(0, 0), (0, pad)], constant_values=0)
                t = t + pad
            x = waveform.reshape((bsz, t // p, p, 1))
            return x

        def __call__(self, waveform: mx.array) -> tuple[mx.array, list[mx.array]]:
            x = self._reshape_period(waveform)
            features: list[mx.array] = []
            for layer in self.layers:
                x = layer(x)
                if isinstance(layer, nn.Conv2d):
                    features.append(x)
            logits = x
            return logits, features

    def __init__(self, periods: list[int] = [2, 3, 5, 7, 11]):
        super().__init__()
        self.helpers = [self.Helper(period=p) for p in periods]

    def __call__(self, waveform: mx.array) -> list[dict[str, object]]:
        waveform = ensure_waveform_2d(waveform)
        outputs: list[dict[str, object]] = []
        for helper in self.helpers:
            logits, features = helper(waveform)
            outputs.append(
                {
                    "period": helper.period,
                    "logits": logits,
                    "features": features,
                }
            )
        return outputs
