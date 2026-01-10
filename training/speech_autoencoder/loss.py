import mlx.core as mx
import mlx.nn as nn

from networks.speech_autoencoder.model import SpeechAutoEncoder
from training.speech_autoencoder.discriminators import MPD, MRD
from training.speech_autoencoder.mels import MelSpectrogramConfig, MelSpectrogramEncoder
from training.speech_autoencoder.utils import ensure_waveform_2d, hann_window

LAMBDA_RECON = 45.0
LAMBDA_ADV = 1.0
LAMBDA_FM = 0.1
ADV_KIND = "ls"  # "hinge" | "lsgan" | "ls"


def frame_signal_1d(x: mx.array, frame_length: int, hop_length: int) -> mx.array:
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


def stft_magnitude(
    waveform: mx.array,
    fft_size: int,
    hop_length: int,
    win_length: int,
    power: float = 2.0,
) -> mx.array:
    waveform = ensure_waveform_2d(waveform)
    pad = max((fft_size - hop_length) // 2, 0)
    if pad > 0:
        waveform = mx.pad(waveform, pad_width=[(0, 0), (pad, pad)], constant_values=0)

    window = hann_window(win_length, dtype=mx.float32)
    frames = frame_signal_1d(waveform, frame_length=win_length, hop_length=hop_length)
    frames = frames * window[None, None, :]

    if win_length < fft_size:
        frames = mx.pad(
            frames,
            pad_width=[(0, 0), (0, 0), (0, fft_size - win_length)],
            constant_values=0,
        )

    spec = mx.fft.rfft(frames, n=fft_size, axis=-1)
    mag = mx.abs(spec)
    if power != 1.0:
        mag = mag**power
    return mag


def mel_from_stft_mag(
    mag: mx.array,
    fft_size: int,
    sample_rate: int,
    n_mels: int,
    f_min: float,
    f_max: float | None,
    log_mel: bool,
) -> mx.array:
    config = MelSpectrogramConfig(
        sample_rate=sample_rate,
        n_fft=fft_size,
        hop_length=max(fft_size // 4, 1),
        win_length=fft_size,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
        power=2.0,
    )
    mel_filter = MelSpectrogramEncoder(config).mel_filter  # (n_mels, freq_bins) numpy
    mel_filter_mx = mx.array(mel_filter.T.astype("float32"))  # (freq_bins, n_mels)

    mel = mag @ mel_filter_mx
    mel = mx.maximum(mel, mx.array(1e-7, dtype=mel.dtype))
    if log_mel:
        mel = mx.log(mel)
    return mel


def ensure_logits(outputs: object, arg_name: str) -> list[mx.array]:
    if isinstance(outputs, mx.array):
        return [outputs]
    if isinstance(outputs, list):
        logits: list[mx.array] = []
        for idx, item in enumerate(outputs):
            if not isinstance(item, dict) or "logits" not in item:
                raise TypeError(f"{arg_name}[{idx}] must be a dict with a 'logits' key")
            logits.append(item["logits"])  # type: ignore[index]
        return logits
    raise TypeError(f"{arg_name} must be an mx.array or list[dict]")


def reconstruction_loss(
    generated_waveform: mx.array,
    target_waveform: mx.array,
    fft_sizes: tuple[int, ...] = (768, 1536, 3072),
    sample_rate: int = 32_000,
    n_mels: tuple[int, ...] | int = (64, 128, 128),
    f_min: float = 0.0,
    f_max: float | None = None,
    log_mel: bool = True,
) -> mx.array:
    losses: list[mx.array] = []
    if isinstance(n_mels, int):
        n_mels_per_fft = [n_mels] * len(fft_sizes)
    else:
        if len(n_mels) != len(fft_sizes):
            raise ValueError("n_mels must match fft_sizes in length")
        n_mels_per_fft = n_mels

    for fft_size, n_mel in zip(fft_sizes, n_mels_per_fft, strict=True):
        hop = max(fft_size // 4, 1)
        mag_gen = stft_magnitude(
            generated_waveform,
            fft_size=fft_size,
            hop_length=hop,
            win_length=fft_size,
            power=2.0,
        )
        mag_tgt = stft_magnitude(
            target_waveform,
            fft_size=fft_size,
            hop_length=hop,
            win_length=fft_size,
            power=2.0,
        )
        mel_gen = mel_from_stft_mag(
            mag_gen,
            fft_size=fft_size,
            sample_rate=sample_rate,
            n_mels=n_mel,
            f_min=f_min,
            f_max=f_max,
            log_mel=log_mel,
        )
        mel_tgt = mel_from_stft_mag(
            mag_tgt,
            fft_size=fft_size,
            sample_rate=sample_rate,
            n_mels=n_mel,
            f_min=f_min,
            f_max=f_max,
            log_mel=log_mel,
        )
        losses.append(mx.mean(mx.abs(mel_gen - mel_tgt)))

    return mx.mean(mx.stack(losses))


def discriminator_adversarial_loss(
    real_outputs: object,
    fake_outputs: object,
    kind: str = "hinge",
) -> mx.array:
    real_logits = ensure_logits(real_outputs, "real_outputs")
    fake_logits = ensure_logits(fake_outputs, "fake_outputs")

    losses: list[mx.array] = []
    for real, fake in zip(real_logits, fake_logits, strict=True):
        if kind == "hinge":
            losses.append(mx.mean(nn.relu(1.0 - real)) + mx.mean(nn.relu(1.0 + fake)))
        elif kind in {"lsgan", "ls"}:
            losses.append(mx.mean((real - 1.0) ** 2) + mx.mean((fake + 1.0) ** 2))
        else:
            raise ValueError(f"Unknown adversarial loss kind: {kind}")
    return mx.mean(mx.stack(losses))


def adversarial_loss(fake_outputs: object, kind: str = "hinge") -> mx.array:
    fake_logits = ensure_logits(fake_outputs, "fake_outputs")

    losses: list[mx.array] = []
    for fake in fake_logits:
        if kind == "hinge":
            losses.append(-mx.mean(fake))
        elif kind in {"lsgan", "ls"}:
            losses.append(mx.mean((fake - 1.0) ** 2))
        else:
            raise ValueError(f"Unknown adversarial loss kind: {kind}")
    return mx.mean(mx.stack(losses))


def feature_matching_loss(
    real_outputs: list[dict[str, object]],
    fake_outputs: list[dict[str, object]],
) -> mx.array:
    losses: list[mx.array] = []
    for real, fake in zip(real_outputs, fake_outputs, strict=True):
        real_feats = real.get("features")
        fake_feats = fake.get("features")

        if not isinstance(real_feats, list) or not isinstance(fake_feats, list):
            raise TypeError("Expected 'features' to be a list[mx.array] in discriminator outputs")

        for r, f in zip(real_feats, fake_feats, strict=True):
            losses.append(mx.mean(mx.abs(r - f)))

    return mx.mean(mx.stack(losses)) if losses else mx.array(0.0, dtype=mx.float32)


def mrd_loss_fn(model: MRD, real_waveform: mx.array, fake_waveform: mx.array) -> mx.array:
    fake = mx.stop_gradient(fake_waveform)
    return discriminator_adversarial_loss(model(real_waveform), model(fake), kind=ADV_KIND)


def mpd_loss_fn(model: MPD, real_waveform: mx.array, fake_waveform: mx.array) -> mx.array:
    fake = mx.stop_gradient(fake_waveform)
    return discriminator_adversarial_loss(model(real_waveform), model(fake), kind=ADV_KIND)


def g_loss_fn(
    autoencoder: SpeechAutoEncoder,
    mrd: MRD,
    mpd: MPD,
    mel: mx.array,
    waveform: mx.array,
) -> tuple[mx.array, dict]:
    generated = autoencoder(mel)
    l_recon = reconstruction_loss(generated, waveform)

    mrd_fake = mrd(generated)
    mpd_fake = mpd(generated)
    l_adv = 0.5 * (
        adversarial_loss(mrd_fake, kind=ADV_KIND) + adversarial_loss(mpd_fake, kind=ADV_KIND)
    )

    mrd_real = mrd(waveform)
    mpd_real = mpd(waveform)
    l_fm = 0.5 * (
        feature_matching_loss(mrd_real, mrd_fake) + feature_matching_loss(mpd_real, mpd_fake)
    )

    return LAMBDA_RECON * l_recon + LAMBDA_ADV * l_adv + LAMBDA_FM * l_fm, {
        "generated": generated,
        "reconstruction_loss": l_recon,
        "adversarial_loss": l_adv,
        "feature_matching_loss": l_fm,
    }
