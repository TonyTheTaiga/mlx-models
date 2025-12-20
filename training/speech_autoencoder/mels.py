from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

_EPS = np.finfo(np.float32).eps


@dataclass(slots=True)
class MelSpectrogramConfig:
    sample_rate: int = 16_000
    n_fft: int = 1024
    hop_length: int = 256
    win_length: int | None = None
    n_mels: int = 228
    f_min: float = 0.0
    f_max: float | None = None
    power: float = 2.0

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.n_fft <= 0 or self.n_fft % 2 != 0:
            raise ValueError("n_fft must be a positive even integer")
        if self.hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if self.win_length is None:
            self.win_length = self.n_fft
        if self.win_length <= 0 or self.win_length > self.n_fft:
            raise ValueError("win_length must be in (0, n_fft]")
        if self.n_mels <= 0:
            raise ValueError("n_mels must be positive")
        if self.f_max is None:
            self.f_max = self.sample_rate / 2
        if not (0.0 <= self.f_min < self.f_max <= self.sample_rate / 2):
            raise ValueError("f_min/f_max must lie within [0, sample_rate / 2]")
        if self.power <= 0:
            raise ValueError("power must be positive")


def hz_to_mel(freq: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(freq) / 700.0)


def mel_to_hz(mels: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mels) / 2595.0) - 1.0)


def build_mel_filter(config: MelSpectrogramConfig) -> np.ndarray:
    freq_bins = config.n_fft // 2 + 1
    mel_min = hz_to_mel(config.f_min)
    mel_max = hz_to_mel(config.f_max)
    mel_points = np.linspace(mel_min, mel_max, config.n_mels + 2, dtype=np.float32)
    hz_points = mel_to_hz(mel_points)
    bin_indices = np.floor((config.n_fft + 1) * hz_points / config.sample_rate).astype(int)
    filter_bank = np.zeros((config.n_mels, freq_bins), dtype=np.float32)

    for idx in range(config.n_mels):
        left = int(bin_indices[idx])
        center = int(bin_indices[idx + 1])
        right = int(bin_indices[idx + 2])

        left = np.clip(left, 0, freq_bins - 1)
        center = np.clip(center, left + 1, freq_bins - 1)
        right = np.clip(right, center + 1, freq_bins)

        up_den = max(center - left, 1)
        down_den = max(right - center, 1)

        if center > left:
            up = (np.arange(center - left, dtype=np.float32) + 1) / up_den
            filter_bank[idx, left:center] = up
        filter_bank[idx, center] = 1.0
        if right > center + 1:
            down = (np.arange(right - center - 1, dtype=np.float32)[::-1] + 1) / down_den
            filter_bank[idx, center + 1 : right] = down

        norm = np.sum(filter_bank[idx])
        if norm > 0:
            filter_bank[idx] /= norm

    return filter_bank


def ensure_numpy(array: Sequence[float] | np.ndarray) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return array
    if hasattr(array, "numpy"):
        return np.asarray(array.numpy())
    return np.asarray(array)


def prepare_signal(samples: Sequence[float] | np.ndarray) -> np.ndarray:
    audio = ensure_numpy(samples).astype(np.float32).flatten()
    if audio.size == 0:
        raise ValueError("waveform must contain at least one sample")
    return np.ascontiguousarray(audio)


def pad_signal(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int,
    target_num_frames: int | None,
) -> np.ndarray:
    if target_num_frames is None:
        if signal.size < frame_length:
            pad = frame_length - signal.size
        else:
            remainder = (signal.size - frame_length) % hop_length
            pad = 0 if remainder == 0 else hop_length - remainder
        if pad > 0:
            signal = np.pad(signal, (0, pad), mode="constant")
        return signal

    total = hop_length * (target_num_frames - 1) + frame_length
    if signal.size < total:
        signal = np.pad(signal, (0, total - signal.size), mode="constant")
    else:
        signal = signal[:total]
    return signal


def frame_signal(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int,
    target_num_frames: int | None,
) -> np.ndarray:
    padded = pad_signal(signal, frame_length, hop_length, target_num_frames)
    if padded.size < frame_length:
        padded = np.pad(padded, (0, frame_length - padded.size), mode="constant")
    num_frames = (
        target_num_frames
        if target_num_frames is not None
        else 1 + (padded.size - frame_length) // hop_length
    )
    shape = (num_frames, frame_length)
    strides = (hop_length * padded.strides[0], padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(padded, shape=shape, strides=strides)
    return np.array(frames, copy=True)


def stft(
    audio: np.ndarray,
    config: MelSpectrogramConfig,
    window: np.ndarray,
    target_num_frames: int | None = None,
) -> np.ndarray:
    frames = frame_signal(audio, config.win_length, config.hop_length, target_num_frames)
    frames *= window
    if config.win_length < config.n_fft:
        frames = np.pad(frames, ((0, 0), (0, config.n_fft - config.win_length)))
    return np.fft.rfft(frames, n=config.n_fft, axis=-1)


def istft(stft_matrix: np.ndarray, config: MelSpectrogramConfig, window: np.ndarray) -> np.ndarray:
    frames = np.fft.irfft(stft_matrix, n=config.n_fft, axis=-1)
    frames = frames[:, : config.win_length]
    frames *= window

    num_frames = frames.shape[0]
    signal_length = config.hop_length * (num_frames - 1) + config.win_length
    signal = np.zeros(signal_length, dtype=np.float32)
    window_acc = np.zeros(signal_length, dtype=np.float32)
    window_square = window**2

    for idx, frame in enumerate(frames):
        start = idx * config.hop_length
        end = start + config.win_length
        signal[start:end] += frame
        window_acc[start:end] += window_square

    non_zero = window_acc > 1e-6
    signal[non_zero] /= window_acc[non_zero]
    return signal


def griffin_lim(
    magnitude: np.ndarray,
    config: MelSpectrogramConfig,
    window: np.ndarray,
    iterations: int,
) -> np.ndarray:
    rng = np.random.default_rng()
    phases = np.exp(2j * np.pi * rng.random(magnitude.shape, dtype=np.float32))
    complex_spec = magnitude * phases

    for _ in range(max(iterations, 1)):
        waveform = istft(complex_spec, config, window)
        stft_matrix = stft(
            waveform,
            config,
            window,
            target_num_frames=magnitude.shape[0],
        )
        angles = np.exp(1j * np.angle(stft_matrix))
        complex_spec = magnitude * angles

    return istft(complex_spec, config, window)


class MelSpectrogramEncoder:
    def __init__(self, config: MelSpectrogramConfig | None = None):
        self.config = config or MelSpectrogramConfig()
        self._window = np.hanning(self.config.win_length).astype(np.float32)
        if np.allclose(self._window.sum(), 0.0):
            self._window[:] = 1.0
        self._mel_filter = build_mel_filter(self.config)

    @property
    def mel_filter(self) -> np.ndarray:
        return self._mel_filter

    @property
    def window(self) -> np.ndarray:
        return self._window

    def encode(
        self,
        waveform: Sequence[float] | np.ndarray,
        log_mel: bool = True,
        pad_mode: str | None = "reflect",
        as_mx: bool = False,
    ):
        audio = prepare_signal(waveform)
        if pad_mode is not None:
            pad = max((self.config.n_fft - self.config.hop_length) // 2, 0)
            if pad > 0:
                mode = pad_mode
                if mode == "reflect" and audio.size <= 1:
                    mode = "constant"
                audio = np.pad(audio, (pad, pad), mode=mode)  # pyright: ignore

        stft_matrix = stft(audio, self.config, self._window)
        magnitude = np.abs(stft_matrix) ** self.config.power
        mel = magnitude @ self._mel_filter.T
        mel = np.maximum(mel, _EPS)
        if log_mel:
            mel = np.log(mel)
        mel = mel.astype(np.float32)
        if as_mx:
            import mlx.core as mx

            return mx.array(mel, dtype=mx.float32)
        return mel

    __call__ = encode


class MelSpectrogramDecoder:
    def __init__(
        self,
        encoder: MelSpectrogramEncoder | None = None,
        griffin_lim_iterations: int = 32,
    ):
        self.encoder = encoder or MelSpectrogramEncoder()
        self.config = self.encoder.config
        self._window = self.encoder.window
        self._mel_filter = self.encoder.mel_filter
        self._mel_filter_pinv_T = np.linalg.pinv(self._mel_filter).T.astype(np.float32)
        self.griffin_lim_iterations = griffin_lim_iterations

    def decode(
        self,
        mel_spectrogram: Sequence[Sequence[float]] | np.ndarray,
        log_mel: bool = True,
        as_mx: bool = False,
    ):
        mel = ensure_numpy(mel_spectrogram).astype(np.float32)
        if mel.ndim != 2:
            raise ValueError("mel_spectrogram must be a 2D array of shape (frames, bins)")
        if log_mel:
            mel = np.exp(mel)
        power_spec = mel @ self._mel_filter_pinv_T
        power_spec = np.maximum(power_spec, _EPS)
        magnitude = power_spec ** (1.0 / self.config.power)

        waveform = griffin_lim(
            magnitude,
            self.config,
            self._window,
            iterations=self.griffin_lim_iterations,
        )

        if as_mx:
            import mlx.core as mx

            return mx.array(waveform, dtype=mx.float32)
        return waveform

    __call__ = decode
