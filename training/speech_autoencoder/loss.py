import mlx.core as mx

from networks.speech_autoencoder.model import SpeechAutoEncoder
from training.speech_autoencoder.discriminators import MPD, MRD
from training.speech_autoencoder.mels import MelSpectrogramConfig, MelSpectrogramEncoder
from training.speech_autoencoder.utils import ensure_waveform_2d, hann_window

LAMBDA_RECON = 45.0
LAMBDA_ADV = 1.0
LAMBDA_FM = 0.1


def frame_signal_1d(x: mx.array, frame_length: int, hop_length: int) -> mx.array:
    t = x.shape[1]
    remainder = (t - frame_length) % hop_length
    pad = 0 if remainder == 0 else hop_length - remainder
    if t < frame_length:
        pad = frame_length - t
    if pad:
        x = mx.pad(x, pad_width=[(0, 0), (0, pad)], constant_values=0)

    num_frames = 1 + (x.shape[1] - frame_length) // hop_length
    base = mx.arange(num_frames, dtype=mx.int32)[:, None] * hop_length
    offsets = mx.arange(frame_length, dtype=mx.int32)[None, :]
    indices = (base + offsets).reshape((-1,))
    return mx.take(x, indices, axis=1).reshape((x.shape[0], num_frames, frame_length))


def stft_magnitude(waveform: mx.array, fft_size: int) -> mx.array:
    waveform = ensure_waveform_2d(waveform)
    hop_length = fft_size // 4
    pad = (fft_size - hop_length) // 2
    if pad > 0:
        waveform = mx.pad(waveform, pad_width=[(0, 0), (pad, pad)], constant_values=0)

    window = hann_window(fft_size, dtype=mx.float32)
    frames = frame_signal_1d(waveform, frame_length=fft_size, hop_length=hop_length)
    frames = frames * window[None, None, :]
    spec = mx.fft.rfft(frames, n=fft_size, axis=-1)
    return mx.abs(spec) ** 2


def mel_from_stft_mag(
    mag: mx.array, fft_size: int, sample_rate: int, n_mels: int
) -> mx.array:
    config = MelSpectrogramConfig(
        sample_rate=sample_rate,
        n_fft=fft_size,
        hop_length=fft_size // 4,
        win_length=fft_size,
        n_mels=n_mels,
    )
    mel_filter = MelSpectrogramEncoder(config).mel_filter
    mel_filter_mx = mx.array(mel_filter.T.astype("float32"))
    mel = mag @ mel_filter_mx
    return mx.log(mx.maximum(mel, 1e-7))


def reconstruction_loss(
    generated: mx.array,
    target: mx.array,
    fft_sizes: tuple[int, ...] = (768, 1536, 3072),
    n_mels: tuple[int, ...] = (64, 128, 128),
    sample_rate: int = 32_000,
) -> mx.array:
    losses = []
    for fft_size, n_mel in zip(fft_sizes, n_mels):
        mel_gen = mel_from_stft_mag(stft_magnitude(generated, fft_size), fft_size, sample_rate, n_mel)
        mel_tgt = mel_from_stft_mag(stft_magnitude(target, fft_size), fft_size, sample_rate, n_mel)
        losses.append(mx.mean(mx.abs(mel_gen - mel_tgt)))
    return mx.mean(mx.stack(losses))


def discriminator_loss(real_outputs: list, fake_outputs: list) -> mx.array:
    """LSGAN discriminator loss: (D(real) - 1)^2 + (D(fake) + 1)^2"""
    losses = []
    for real, fake in zip(real_outputs, fake_outputs):
        real_logits = real["logits"]
        fake_logits = fake["logits"]
        losses.append(mx.mean((real_logits - 1.0) ** 2) + mx.mean((fake_logits + 1.0) ** 2))
    return mx.mean(mx.stack(losses))


def generator_adversarial_loss(fake_outputs: list) -> mx.array:
    """LSGAN generator loss: (D(fake) - 1)^2"""
    losses = []
    for fake in fake_outputs:
        losses.append(mx.mean((fake["logits"] - 1.0) ** 2))
    return mx.mean(mx.stack(losses))


def feature_matching_loss(real_outputs: list, fake_outputs: list) -> mx.array:
    total = mx.array(0.0)
    for real, fake in zip(real_outputs, fake_outputs):
        for r, f in zip(real["features"], fake["features"]):
            total = total + mx.mean(mx.abs(r - f))
    return total


def mrd_loss_fn(model: MRD, real_waveform: mx.array, fake_waveform: mx.array) -> mx.array:
    return discriminator_loss(model(real_waveform), model(fake_waveform))


def mpd_loss_fn(model: MPD, real_waveform: mx.array, fake_waveform: mx.array) -> mx.array:
    return discriminator_loss(model(real_waveform), model(fake_waveform))


def g_loss_fn(
    autoencoder: SpeechAutoEncoder,
    mrd: MRD,
    mpd: MPD,
    mel: mx.array,
    waveform: mx.array,
) -> tuple[mx.array, dict]:
    generated = autoencoder(mel)

    # Run discriminators
    mrd_fake, mpd_fake = mrd(generated), mpd(generated)
    mrd_real, mpd_real = mrd(waveform), mpd(waveform)
    all_fake, all_real = mrd_fake + mpd_fake, mrd_real + mpd_real

    # Compute losses
    l_recon = reconstruction_loss(generated, waveform)
    l_adv = generator_adversarial_loss(all_fake)
    l_fm = feature_matching_loss(all_real, all_fake)

    total = LAMBDA_RECON * l_recon + LAMBDA_ADV * l_adv + LAMBDA_FM * l_fm
    return total, {
        "generated": generated,
        "reconstruction_loss": l_recon,
        "adversarial_loss": l_adv,
        "feature_matching_loss": l_fm,
    }
