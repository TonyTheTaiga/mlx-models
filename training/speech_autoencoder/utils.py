import math
import wave
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from training.speech_autoencoder.mels import MelSpectrogramEncoder

if TYPE_CHECKING:
    from networks.speech_autoencoder.model import SpeechAutoEncoder
    from training.speech_autoencoder.dataset import SpsCorpusDataset
    from training.speech_autoencoder.mels import MelSpectrogramConfig


def ensure_waveform_2d(waveform: mx.array) -> mx.array:
    if waveform.ndim == 1:
        waveform = waveform[None, :]
    if waveform.ndim == 3 and waveform.shape[-1] == 1:
        waveform = waveform[..., 0]
    if waveform.ndim != 2:
        raise ValueError(
            "Expected waveform shaped (B, T) or (B, T, 1) (or (T,) for a single example)."
        )
    return waveform.astype(mx.float32)


def hann_window(length: int, dtype: mx.Dtype = mx.float32) -> mx.array:
    if length <= 1:
        return mx.ones((length,), dtype=dtype)
    n = mx.arange(length, dtype=dtype)
    return 0.5 - 0.5 * mx.cos((2.0 * mx.array(math.pi, dtype=dtype) * n) / (length - 1))


def reflect_pad_1d(x: mx.array, pad_left: int, pad_right: int, axis: int = -1) -> mx.array:
    if pad_left < 0 or pad_right < 0:
        raise ValueError("pad_left and pad_right must be non-negative")
    if pad_left == 0 and pad_right == 0:
        return x

    axis = axis if axis >= 0 else x.ndim + axis
    if axis < 0 or axis >= x.ndim:
        raise ValueError("axis out of range")

    n = int(x.shape[axis])
    if n <= 0:
        raise ValueError("Cannot pad an empty axis")
    if n == 1:
        pad_width = [(0, 0)] * x.ndim
        pad_width[axis] = (pad_left, pad_right)
        return mx.pad(x, pad_width=pad_width, constant_values=float(x.reshape((-1,))[0].item()))

    period = 2 * (n - 1)
    positions = mx.arange(-pad_left, n + pad_right, dtype=mx.int32)
    pos_mod = mx.remainder(positions, period)
    indices = mx.where(pos_mod <= (n - 1), pos_mod, period - pos_mod)
    return mx.take(x, indices, axis=axis)


def write_wav_mono(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32).flatten()
    audio = np.clip(audio, -1.0, 1.0)
    pcm16 = (audio * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(int(sample_rate))
        f.writeframes(pcm16.tobytes())


def waveform_stats(audio: np.ndarray) -> dict[str, float]:
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.size == 0:
        return {"peak": 0.0, "rms": 0.0, "clip_frac": 0.0}
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(audio**2)))
    clip_frac = float(np.mean(np.abs(audio) > 1.0))
    return {"peak": peak, "rms": rms, "clip_frac": clip_frac}


def peak_normalize(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    current = float(np.max(np.abs(audio))) if audio.size else 0.0
    if current <= 0:
        return audio
    return audio * (peak / current)


def reconstruct_mel_in_chunks(
    model: "SpeechAutoEncoder",
    mel: mx.array,
    out_dims: int,
    chunk_frames: int = 256,
    overlap_frames: int = 128,
) -> mx.array:
    """Run the autoencoder over a long mel sequence with overlap-add.

    Uses Hann windowing and overlap-add for smooth chunk boundaries.
    For mathematically perfect reconstruction, use 50% overlap (overlap_frames = chunk_frames // 2).
    Returns waveform shaped (B, T, 1).
    """
    if mel.ndim != 3:
        raise ValueError("Expected mel shaped (B, frames, bins)")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    if overlap_frames < 0 or overlap_frames >= chunk_frames:
        raise ValueError("overlap_frames must be in [0, chunk_frames)")
    if out_dims <= 0:
        raise ValueError("out_dims must be positive")

    bsz, total_frames, _bins = mel.shape

    # Handle edge case: audio shorter than one chunk
    if total_frames <= chunk_frames:
        wav = model(mel)
        mx.eval(wav)
        return wav

    hop_frames = chunk_frames - overlap_frames
    total_samples = total_frames * out_dims
    chunk_samples = chunk_frames * out_dims

    # Output buffers (numpy for easy in-place accumulation)
    output = np.zeros((bsz, total_samples, 1), dtype=np.float32)
    weight_sum = np.zeros((1, total_samples, 1), dtype=np.float32)

    # Create Hann window in sample domain
    window_full = np.hanning(chunk_samples).astype(np.float32)

    start = 0
    while start < total_frames:
        end = min(start + chunk_frames, total_frames)
        actual_chunk_frames = end - start
        actual_samples = actual_chunk_frames * out_dims

        mel_chunk = mel[:, start:end, :]
        wav_chunk = model(mel_chunk)
        mx.eval(wav_chunk)
        wav_np = np.asarray(wav_chunk)

        sample_start = start * out_dims
        sample_end = sample_start + actual_samples

        # Get appropriate window (shorter window for last chunk if needed)
        if actual_chunk_frames == chunk_frames:
            window = window_full
        else:
            window = np.hanning(actual_samples).astype(np.float32)

        # Apply window and accumulate
        window_3d = window[None, :, None]
        output[:, sample_start:sample_end, :] += wav_np * window_3d
        weight_sum[:, sample_start:sample_end, :] += window_3d

        start += hop_frames

    # Normalize by weight sum (avoid division by zero)
    weight_sum = np.maximum(weight_sum, 1e-8)
    output = output / weight_sum

    return mx.array(output, dtype=mx.float32)


def save_full_reconstruction(
    dataset: "SpsCorpusDataset",
    model: "SpeechAutoEncoder",
    mel_cfg: "MelSpectrogramConfig",
    sample_rate: int,
    out_dir: Path,
    sample_index: int = 0,
    chunk_frames: int = 256,
    overlap_frames: int = 128,
) -> None:
    sample = dataset[sample_index]
    wav, _sr = sample.load_waveform(target_sr=sample_rate, as_mx=False)
    wav = np.asarray(wav, dtype=np.float32)
    orig_len = int(wav.shape[0])

    hop = int(mel_cfg.hop_length)
    aligned_len = int(math.ceil(orig_len / hop) * hop)
    if aligned_len != orig_len:
        wav_padded = np.pad(wav, (0, aligned_len - orig_len), mode="constant")
    else:
        wav_padded = wav

    encoder = MelSpectrogramEncoder(mel_cfg)
    mel = encoder.encode(
        wav_padded,
        log_mel=True,
        pad_mode="reflect",
        as_mx=False,
    )
    mel_mx = mx.array(mel[None, :, :], dtype=mx.float32)

    model.train(False)
    try:
        fake = reconstruct_mel_in_chunks(
            model,
            mel_mx,
            out_dims=hop,
            chunk_frames=chunk_frames,
            overlap_frames=overlap_frames,
        )
        mx.eval(fake)
    finally:
        model.train(True)

    fake_np = np.asarray(fake)[0, :orig_len, 0]
    real_stats = waveform_stats(wav)
    fake_stats = waveform_stats(fake_np)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_path = out_dir / f"full_{sample.audio_id}_stats.txt"
    stats_path.write_text(
        "real "
        + " ".join(f"{k}={v:.6f}" for k, v in real_stats.items())
        + "\n"
        + "fake "
        + " ".join(f"{k}={v:.6f}" for k, v in fake_stats.items())
        + "\n",
        encoding="utf-8",
    )

    fake_to_write = fake_np
    if fake_stats["peak"] > 1.0:
        fake_to_write = peak_normalize(fake_to_write, peak=0.95)

    write_wav_mono(out_dir / f"full_{sample.audio_id}_real.wav", wav, sample_rate)
    write_wav_mono(out_dir / f"full_{sample.audio_id}_fake.wav", fake_to_write, sample_rate)

    mel_fake = encoder.encode(fake_np, log_mel=True, pad_mode="reflect", as_mx=False)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    im0 = axes[0].imshow(
        mel.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
    )
    axes[0].set_title(f"Original Mel Spectrogram (ID: {sample.audio_id})")
    axes[0].set_xlabel("Frame")
    axes[0].set_ylabel("Mel Bin")
    plt.colorbar(im0, ax=axes[0], label="Log Magnitude")

    im1 = axes[1].imshow(
        mel_fake.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
    )
    axes[1].set_title("Reconstructed Mel Spectrogram")
    axes[1].set_xlabel("Frame")
    axes[1].set_ylabel("Mel Bin")
    plt.colorbar(im1, ax=axes[1], label="Log Magnitude")

    plt.tight_layout()
    plt.savefig(out_dir / f"full_{sample.audio_id}_spectrogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    def check(shape, pad_left, pad_right, axis):
        x_np = rng.normal(size=shape).astype(np.float32)
        x_mx = mx.array(x_np)

        y_mx = reflect_pad_1d(x_mx, pad_left, pad_right, axis=axis)
        y_mx_np = np.asarray(y_mx)

        pad_width = [(0, 0)] * len(shape)
        pad_width[axis if axis >= 0 else len(shape) + axis] = (pad_left, pad_right)
        y_np = np.pad(x_np, pad_width, mode="reflect")

        max_abs = float(np.max(np.abs(y_mx_np - y_np)))
        assert np.allclose(y_mx_np, y_np, atol=1e-6, rtol=0.0), (
            f"mismatch shape={shape} axis={axis} pad=({pad_left},{pad_right}) max_abs={max_abs}"
        )

    check((10,), 3, 4, -1)
    check((2,), 1, 1, -1)
    check((3, 10), 4, 2, -1)
    check((10, 3), 2, 5, 0)
    check((2, 13, 1), 7, 3, 1)

    print("reflect_pad_1d smoke tests passed")
