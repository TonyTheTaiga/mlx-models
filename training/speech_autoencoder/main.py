import argparse
import math
import random
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import tqdm

from networks.speech_autoencoder.model import SpeechAutoEncoder
from training.speech_autoencoder.dataset import SpsCorpusDataset
from training.speech_autoencoder.discriminators import MPD, MRD
from training.speech_autoencoder.loss import (
    ADV_KIND,
    LAMBDA_ADV,
    LAMBDA_FM,
    LAMBDA_RECON,
    adversarial_loss,
    feature_matching_loss,
    loss_fn,
    mpd_loss_fn,
    mrd_loss_fn,
    reconstruction_loss,
)
from training.speech_autoencoder.mels import MelSpectrogramConfig, MelSpectrogramEncoder
from training.speech_autoencoder.utils import save_full_reconstruction


def _compute_learning_rate(
    step: int,
    *,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    total_steps: int,
    schedule: str,
) -> float:
    if base_lr <= 0:
        raise ValueError("base_lr must be positive")
    if min_lr < 0:
        raise ValueError("min_lr must be non-negative")
    if schedule != "none" and min_lr > base_lr:
        raise ValueError("min_lr must be <= base_lr when using decay")
    if warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")

    step = int(step)
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)

    if schedule == "none":
        return base_lr

    t = max(step - warmup_steps, 0)
    T = max(total_steps - warmup_steps - 1, 1)
    frac = min(float(t) / float(T), 1.0)

    if schedule == "cosine":
        cosine = 0.5 * (1.0 + math.cos(math.pi * frac))
        return min_lr + (base_lr - min_lr) * cosine
    if schedule == "linear":
        return base_lr + (min_lr - base_lr) * frac
    if schedule == "exponential":
        if min_lr == 0.0:
            return base_lr * (0.1 ** frac)
        return base_lr * math.exp(math.log(min_lr / base_lr) * frac)

    raise ValueError(f"Unknown lr schedule: {schedule}")


def _set_optim_lr(optimizer: optim.Optimizer, lr: float) -> None:
    optimizer.learning_rate = float(lr)


def loss_and_grads_fn(
    *,
    ae: SpeechAutoEncoder,
    mrd: MRD,
    mpd: MPD,
    mel: mx.array,
    real_waveform: mx.array,
    mrd_loss_and_grad_fn,
    mpd_loss_and_grad_fn,
    g_loss_and_grad_fn,
) -> tuple[dict[str, float], object, object, object]:
    fake_waveform = ae(mel)

    recon = reconstruction_loss(fake_waveform, real_waveform)
    mrd_fake_out = mrd(fake_waveform)
    mpd_fake_out = mpd(fake_waveform)
    mrd_real_out = mrd(real_waveform)
    mpd_real_out = mpd(real_waveform)
    adv_mrd = adversarial_loss(mrd_fake_out)
    adv_mpd = adversarial_loss(mpd_fake_out)
    fm_mrd = feature_matching_loss(mrd_real_out, mrd_fake_out)
    fm_mpd = feature_matching_loss(mpd_real_out, mpd_fake_out)

    mrd_d_loss, mrd_grads = mrd_loss_and_grad_fn(mrd, real_waveform, fake_waveform)
    mpd_d_loss, mpd_grads = mpd_loss_and_grad_fn(mpd, real_waveform, fake_waveform)
    g_loss_value, g_grads = g_loss_and_grad_fn(ae, mrd, mpd, mel, real_waveform)

    metrics = {
        "mrd_d": float(mrd_d_loss.item()),
        "mpd_d": float(mpd_d_loss.item()),
        "g": float(g_loss_value.item()),
        "recon": float(recon.item()),
        "adv_mrd": float(adv_mrd.item()),
        "adv_mpd": float(adv_mpd.item()),
        "adv": float((adv_mrd + adv_mpd).item()),
        "fm_mrd": float(fm_mrd.item()),
        "fm_mpd": float(fm_mpd.item()),
        "fm": float((fm_mrd + fm_mpd).item()),
    }
    return metrics, g_grads, mrd_grads, mpd_grads


def train_step(
    *,
    ae: SpeechAutoEncoder,
    mrd: MRD,
    mpd: MPD,
    opt_g: optim.Optimizer,
    opt_mrd: optim.Optimizer,
    opt_mpd: optim.Optimizer,
    mel: mx.array,
    real_waveform: mx.array,
    mrd_loss_and_grad_fn,
    mpd_loss_and_grad_fn,
    g_loss_and_grad_fn,
) -> dict[str, float]:
    metrics, g_grads, mrd_grads, mpd_grads = loss_and_grads_fn(
        ae=ae,
        mrd=mrd,
        mpd=mpd,
        mel=mel,
        real_waveform=real_waveform,
        mrd_loss_and_grad_fn=mrd_loss_and_grad_fn,
        mpd_loss_and_grad_fn=mpd_loss_and_grad_fn,
        g_loss_and_grad_fn=g_loss_and_grad_fn,
    )

    opt_mrd.update(mrd, mrd_grads)
    opt_mpd.update(mpd, mpd_grads)
    opt_g.update(ae, g_grads)

    mx.eval(
        ae.parameters(),
        mrd.parameters(),
        mpd.parameters(),
        opt_g.state,
        opt_mrd.state,
        opt_mpd.state,
    )

    return metrics


def _pad_or_crop_1d(waveform, target_len: int, *, rng: random.Random) -> object:
    if waveform.shape[0] == target_len:
        return waveform
    if waveform.shape[0] < target_len:
        pad = target_len - waveform.shape[0]
        return np.pad(waveform, (0, pad), mode="constant")
    start = rng.randrange(0, waveform.shape[0] - target_len + 1)
    return waveform[start : start + target_len]


def dataloader(
    dataset: SpsCorpusDataset,
    *,
    batch_size: int,
    segment_seconds: float = 1.0,
    sample_rate: int = 32_000,
    mel_config: MelSpectrogramConfig,
    out_dims: int | None = None,
    log_mel: bool = True,
    shuffle: bool = True,
    seed: int = 0,
) -> Iterator[dict[str, mx.array]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive")

    target_len = int(round(segment_seconds * sample_rate))
    if target_len <= 0:
        raise ValueError("segment_seconds * sample_rate must be positive")

    encoder = MelSpectrogramEncoder(mel_config)
    out_dims_val = int(out_dims) if out_dims is not None else int(mel_config.hop_length)
    if out_dims_val <= 0:
        raise ValueError("out_dims must be positive")

    rng = random.Random(seed)
    batch_starts = list(range(0, len(dataset), batch_size))
    if shuffle:
        perm = np.random.default_rng(seed).permutation(len(batch_starts))
        batch_starts = [batch_starts[i] for i in perm]

    batch_mels: list[mx.array] = []
    batch_wavs: list[mx.array] = []
    batch_ids: list[int] = []

    for start in batch_starts:
        samples = dataset[start : start + batch_size]
        for sample in samples:
            wav, _sr = sample.load_waveform(target_sr=sample_rate, as_mx=False)
            aligned_len = max(int(math.ceil(target_len / out_dims_val) * out_dims_val), out_dims_val)
            wav = _pad_or_crop_1d(wav, aligned_len, rng=rng)
            mel = encoder.encode(wav, log_mel=log_mel, as_mx=False)
            if int(mel.shape[0]) * out_dims_val != wav.shape[0]:
                raise ValueError(
                    f"Waveform/mel misalignment: wav={wav.shape[0]} "
                    f"mel_frames={mel.shape[0]} out_dims={out_dims_val}"
                )

            batch_mels.append(mx.array(mel, dtype=mx.float32))
            batch_wavs.append(mx.array(np.asarray(wav), dtype=mx.float32)[:, None])
            batch_ids.append(sample.audio_id)

        if batch_mels:
            yield {
                "mel": mx.stack(batch_mels, axis=0),
                "waveform": mx.stack(batch_wavs, axis=0),
                "audio_id": mx.array(batch_ids, dtype=mx.int32),
            }
            batch_mels = []
            batch_wavs = []
            batch_ids = []


def random_dataloader(
    dataset: SpsCorpusDataset,
    *,
    batch_size: int,
    segment_seconds: float,
    sample_rate: int,
    mel_config: MelSpectrogramConfig,
    out_dims: int | None = None,
    log_mel: bool = True,
    seed: int = 0,
) -> Iterator[dict[str, mx.array]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if segment_seconds <= 0:
        raise ValueError("segment_seconds must be positive")

    target_len = int(round(segment_seconds * sample_rate))
    if target_len <= 0:
        raise ValueError("segment_seconds * sample_rate must be positive")

    encoder = MelSpectrogramEncoder(mel_config)
    out_dims_val = int(out_dims) if out_dims is not None else int(mel_config.hop_length)
    if out_dims_val <= 0:
        raise ValueError("out_dims must be positive")

    aligned_len = max(int(math.ceil(target_len / out_dims_val) * out_dims_val), out_dims_val)

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    while True:
        if len(dataset) == 0:
            raise ValueError("dataset is empty")
        indices = np_rng.integers(0, len(dataset), size=batch_size)
        batch_mels: list[mx.array] = []
        batch_wavs: list[mx.array] = []
        batch_ids: list[int] = []

        for idx in indices:
            sample = dataset[int(idx)]
            wav, _sr = sample.load_waveform(target_sr=sample_rate, as_mx=False)
            wav = _pad_or_crop_1d(wav, aligned_len, rng=rng)
            mel = encoder.encode(wav, log_mel=log_mel, as_mx=False)
            if int(mel.shape[0]) * out_dims_val != wav.shape[0]:
                raise ValueError(
                    f"Waveform/mel misalignment: wav={wav.shape[0]} "
                    f"mel_frames={mel.shape[0]} out_dims={out_dims_val}"
                )

            batch_mels.append(mx.array(mel, dtype=mx.float32))
            batch_wavs.append(mx.array(np.asarray(wav), dtype=mx.float32)[:, None])
            batch_ids.append(sample.audio_id)

        yield {
            "mel": mx.stack(batch_mels, axis=0),
            "waveform": mx.stack(batch_wavs, axis=0),
            "audio_id": mx.array(batch_ids, dtype=mx.int32),
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "sps-corpus-1.0-2025-11-25-en",
    )
    parser.add_argument("--sample-rate", type=int, default=32_000)
    parser.add_argument("--segment-seconds", type=float, default=0.19)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50_000, help="Number of training steps.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Base learning rate.")
    parser.add_argument(
        "--lr-schedule",
        choices=("none", "cosine", "linear", "exponential"),
        default="cosine",
        help="Learning-rate schedule applied per training step.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=2e-5,
        help="Lower bound for decayed learning rate.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help="Linear warmup steps before decay.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Write a full reconstruction every N steps (0 disables).",
    )
    return parser.parse_args()


def run_train(args: argparse.Namespace) -> None:
    dataset = SpsCorpusDataset(dataset_dir=args.dataset_dir, split="train")

    sample_rate = int(args.sample_rate)
    segment_seconds = float(args.segment_seconds)
    batch_size = int(args.batch_size)
    steps = int(args.steps)
    learning_rate = float(args.learning_rate)
    min_learning_rate = float(args.min_learning_rate)
    lr_schedule = str(args.lr_schedule)
    warmup_steps = int(args.warmup_steps)
    seed = int(args.seed)
    log_every = max(1, int(args.log_every))
    save_every = int(args.save_every)

    mel_cfg = MelSpectrogramConfig(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        win_length=2048,
        n_mels=228,
    )

    ae = SpeechAutoEncoder(in_dims=228, hidden_dims=24, out_dims=mel_cfg.hop_length)
    mrd = MRD()
    mpd = MPD()

    opt_g = optim.AdamW(learning_rate=learning_rate)
    opt_mrd = optim.AdamW(learning_rate=learning_rate)
    opt_mpd = optim.AdamW(learning_rate=learning_rate)
    mx.eval(ae.parameters(), mrd.parameters(), mpd.parameters())

    mrd_loss_and_grad_fn = nn.value_and_grad(mrd, mrd_loss_fn)
    mpd_loss_and_grad_fn = nn.value_and_grad(mpd, mpd_loss_fn)
    g_loss_and_grad_fn = nn.value_and_grad(ae, loss_fn)

    total_steps = steps
    print(
        "config:",
        f"sr={sample_rate}",
        f"segment_seconds={segment_seconds:.5f}",
        f"mel_nfft={mel_cfg.n_fft}",
        f"mel_hop={mel_cfg.hop_length}",
        f"mel_nmels={mel_cfg.n_mels}",
        f"lr={learning_rate:g}",
        f"lr_schedule={lr_schedule}",
        f"min_lr={min_learning_rate:g}",
        f"warmup_steps={warmup_steps}",
        f"steps={steps}",
        f"seed={seed}",
        f"lambda_recon={LAMBDA_RECON:g}",
        f"lambda_adv={LAMBDA_ADV:g}",
        f"lambda_fm={LAMBDA_FM:g}",
        f"adv_kind={ADV_KIND}",
    )

    loader = random_dataloader(
        dataset,
        batch_size=batch_size,
        segment_seconds=segment_seconds,
        sample_rate=sample_rate,
        mel_config=mel_cfg,
        out_dims=mel_cfg.hop_length,
        log_mel=True,
        seed=seed,
    )

    sum_mrd_d = 0.0
    sum_mpd_d = 0.0
    sum_g = 0.0
    sum_recon = 0.0
    sum_adv_mrd = 0.0
    sum_adv_mpd = 0.0
    sum_fm_mrd = 0.0
    sum_fm_mpd = 0.0
    num_steps = 0

    window_mrd_d = 0.0
    window_mpd_d = 0.0
    window_g = 0.0
    window_recon = 0.0
    window_adv_mrd = 0.0
    window_adv_mpd = 0.0
    window_fm_mrd = 0.0
    window_fm_mpd = 0.0
    window_count = 0

    progress = tqdm(range(steps), total=steps, desc="train")
    last_lr = learning_rate
    for step in progress:
        lr = _compute_learning_rate(
            step,
            base_lr=learning_rate,
            min_lr=min_learning_rate,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            schedule=lr_schedule,
        )
        last_lr = lr
        _set_optim_lr(opt_g, lr)
        _set_optim_lr(opt_mrd, lr)
        _set_optim_lr(opt_mpd, lr)

        batch = next(loader)
        mel = batch["mel"]
        real_waveform = batch["waveform"]
        metrics = train_step(
            ae=ae,
            mrd=mrd,
            mpd=mpd,
            opt_g=opt_g,
            opt_mrd=opt_mrd,
            opt_mpd=opt_mpd,
            mel=mel,
            real_waveform=real_waveform,
            mrd_loss_and_grad_fn=mrd_loss_and_grad_fn,
            mpd_loss_and_grad_fn=mpd_loss_and_grad_fn,
            g_loss_and_grad_fn=g_loss_and_grad_fn,
        )

        mrd_d_val = metrics["mrd_d"]
        mpd_d_val = metrics["mpd_d"]
        g_val = metrics["g"]

        sum_mrd_d += mrd_d_val
        sum_mpd_d += mpd_d_val
        sum_g += g_val
        sum_recon += metrics["recon"]
        sum_adv_mrd += metrics["adv_mrd"]
        sum_adv_mpd += metrics["adv_mpd"]
        sum_fm_mrd += metrics["fm_mrd"]
        sum_fm_mpd += metrics["fm_mpd"]
        num_steps += 1

        window_mrd_d += mrd_d_val
        window_mpd_d += mpd_d_val
        window_g += g_val
        window_recon += metrics["recon"]
        window_adv_mrd += metrics["adv_mrd"]
        window_adv_mpd += metrics["adv_mpd"]
        window_fm_mrd += metrics["fm_mrd"]
        window_fm_mpd += metrics["fm_mpd"]
        window_count += 1

        progress.set_postfix(
            mrd_d=f"{mrd_d_val:.4f}",
            mpd_d=f"{mpd_d_val:.4f}",
            g=f"{g_val:.4f}",
            recon=f"{metrics['recon']:.4f}",
            adv=f"{metrics['adv']:.4f}",
            fm=f"{metrics['fm']:.4f}",
            lr=f"{lr:.3g}",
        )

        if (step == 0) or ((step + 1) % log_every == 0) or (step + 1 == steps):
            denom = max(window_count, 1)
            print(
                f"step {step + 1}/{steps} "
                f"avg_mrd_d={window_mrd_d / denom:.6f} "
                f"avg_mpd_d={window_mpd_d / denom:.6f} "
                f"avg_g={window_g / denom:.6f} "
                f"avg_recon={window_recon / denom:.6f} "
                f"avg_adv={(window_adv_mrd + window_adv_mpd) / denom:.6f} "
                f"avg_fm={(window_fm_mrd + window_fm_mpd) / denom:.6f} "
                f"lr={last_lr:.6g}"
            )
            window_mrd_d = 0.0
            window_mpd_d = 0.0
            window_g = 0.0
            window_recon = 0.0
            window_adv_mrd = 0.0
            window_adv_mpd = 0.0
            window_fm_mrd = 0.0
            window_fm_mpd = 0.0
            window_count = 0

        if save_every > 0 and (step + 1) % save_every == 0:
            save_full_reconstruction(
                dataset=dataset,
                model=ae,
                mel_cfg=mel_cfg,
                sample_rate=sample_rate,
                out_dir=Path("output") / f"step_{step + 1}",
                sample_index=0,
                chunk_frames=256,
                overlap_frames=32,
            )

    denom = max(num_steps, 1)
    print(
        f"done steps={steps} "
        f"avg_mrd_d={sum_mrd_d / denom:.6f} "
        f"avg_mpd_d={sum_mpd_d / denom:.6f} "
        f"avg_g={sum_g / denom:.6f} "
        f"avg_recon={sum_recon / denom:.6f} "
        f"avg_adv={(sum_adv_mrd + sum_adv_mpd) / denom:.6f} "
        f"avg_fm={(sum_fm_mrd + sum_fm_mpd) / denom:.6f} "
        f"lr={last_lr:.6g}"
    )

    save_full_reconstruction(
        dataset=dataset,
        model=ae,
        mel_cfg=mel_cfg,
        sample_rate=sample_rate,
        out_dir=Path("output") / "final",
        sample_index=0,
        chunk_frames=256,
        overlap_frames=32,
    )


def main() -> None:
    args = _parse_args()
    run_train(args)


if __name__ == "__main__":
    main()
