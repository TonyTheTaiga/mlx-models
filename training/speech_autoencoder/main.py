import argparse
import math
import random
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from tqdm import tqdm
try:
    from tabulate import tabulate as _tabulate
except ImportError:  # pragma: no cover
    _tabulate = None

from networks.speech_autoencoder.model import SpeechAutoEncoder
from training.speech_autoencoder.dataset import SpsCorpusDataset
from training.speech_autoencoder.discriminators import MPD, MRD
from training.speech_autoencoder.loss import (
    g_loss_fn,
    mpd_loss_fn,
    mrd_loss_fn,
)
from training.speech_autoencoder.mels import MelSpectrogramConfig, MelSpectrogramEncoder
from training.speech_autoencoder.utils import save_full_reconstruction


def save_weights(out_dir: Path, ae: SpeechAutoEncoder, mrd: MRD, mpd: MPD) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.eval(ae.parameters(), mrd.parameters(), mpd.parameters())

    arrays: dict[str, np.ndarray] = {}
    for prefix, module in (("ae", ae), ("mrd", mrd), ("mpd", mpd)):
        for key, value in tree_flatten(module.parameters()):
            arrays[f"{prefix}.{key}"] = np.asarray(value)

    np.savez(out_dir / "weights.npz", **arrays)


def compute_learning_rate(
    step: int,
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
            return base_lr * (0.1**frac)
        return base_lr * math.exp(math.log(min_lr / base_lr) * frac)

    raise ValueError(f"Unknown lr schedule: {schedule}")


def set_optim_lr(optimizer: optim.Optimizer, lr: float) -> None:
    optimizer.learning_rate = float(lr)


class MetricLogger:
    def __init__(self) -> None:
        self.total: dict[str, float] = {}
        self.window: dict[str, float] = {}
        self.total_count = 0
        self.window_count = 0
        self.last: dict[str, float] = {}

    def update(self, metrics: dict[str, float]) -> None:
        metrics = {k: float(v) for k, v in metrics.items()}
        self.last = metrics
        self.total_count += 1
        self.window_count += 1
        for key, value in metrics.items():
            self.total[key] = self.total.get(key, 0.0) + value
            self.window[key] = self.window.get(key, 0.0) + value

    def window_mean(self) -> dict[str, float]:
        denom = max(self.window_count, 1)
        return {k: v / denom for k, v in self.window.items()}

    def total_mean(self) -> dict[str, float]:
        denom = max(self.total_count, 1)
        return {k: v / denom for k, v in self.total.items()}

    def reset_window(self) -> None:
        self.window.clear()
        self.window_count = 0


def format_table(rows: list[tuple[str, object]], headers: list[str]) -> str:
    if _tabulate is not None:
        return _tabulate(rows, headers=headers, floatfmt=".6f", tablefmt="plain")

    def _fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    lines = [f"{headers[0]:<18}{headers[1]}"]
    for key, value in rows:
        lines.append(f"{str(key):<18}{_fmt(value)}")
    return "\n".join(lines)


def pad_or_crop_1d(waveform, target_len: int, rng: random.Random) -> object:
    if waveform.shape[0] == target_len:
        return waveform
    if waveform.shape[0] < target_len:
        pad = target_len - waveform.shape[0]
        return np.pad(waveform, (0, pad), mode="constant")
    start = rng.randrange(0, waveform.shape[0] - target_len + 1)
    return waveform[start : start + target_len]


def dataloader(
    dataset: SpsCorpusDataset,
    batch_size: int,
    segment_seconds: float,
    sample_rate: int,
    mel_config: MelSpectrogramConfig,
    log_mel: bool = True,
    seed: int = 0,
) -> Iterator[dict[str, mx.array]]:
    target_len = int(round(segment_seconds * sample_rate))
    encoder = MelSpectrogramEncoder(mel_config)
    out_dims_val = int(mel_config.hop_length)
    aligned_len = max(int(math.ceil(target_len / out_dims_val) * out_dims_val), out_dims_val)
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if len(dataset) == 0:
        raise ValueError("dataset is empty")

    while True:
        indices = np_rng.integers(0, len(dataset), size=batch_size)
        batch_mels: list[mx.array] = []
        batch_wavs: list[mx.array] = []
        batch_ids: list[int] = []

        for idx in indices:
            sample = dataset[int(idx)]
            wav, _sr = sample.load_waveform(target_sr=sample_rate, as_mx=False)
            wav = pad_or_crop_1d(wav, aligned_len, rng=rng)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "sps-corpus-1.0-2025-11-25-en",
    )
    parser.add_argument("--sample-rate", type=int, default=32_000)
    parser.add_argument("--segment-seconds", type=float, default=0.19)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=50_000, help="Number of training steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Base learning rate.")
    parser.add_argument(
        "--disc-learning-rate",
        type=float,
        default=None,
        help="Optional discriminator learning rate (defaults to --learning-rate).",
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("none", "cosine", "linear", "exponential"),
        default="cosine",
        help="Learning-rate schedule applied per training step.",
    )
    parser.add_argument(
        "--min-learning-rate",
        type=float,
        default=1e-5,
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
        default=2500,
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
    disc_learning_rate = (
        float(args.disc_learning_rate) if args.disc_learning_rate is not None else learning_rate
    )
    min_learning_rate = float(args.min_learning_rate)
    lr_schedule = str(args.lr_schedule)
    warmup_steps = int(args.warmup_steps)
    seed = int(args.seed)
    log_every = max(1, int(args.log_every))
    save_every = int(args.save_every)

    mel_cfg = MelSpectrogramConfig(
        sample_rate=sample_rate,
        n_fft=1536,
        hop_length=352,
        win_length=1536,
        n_mels=228,
    )

    ae = SpeechAutoEncoder(in_dims=228, hidden_dims=24, out_dims=mel_cfg.hop_length)
    mrd = MRD()
    mpd = MPD()

    opt_g = optim.AdamW(learning_rate=learning_rate)
    opt_mrd = optim.AdamW(learning_rate=disc_learning_rate)
    opt_mpd = optim.AdamW(learning_rate=disc_learning_rate)
    mx.eval(ae.parameters(), mrd.parameters(), mpd.parameters())
    mrd_loss_and_grad_fn = nn.value_and_grad(mrd, mrd_loss_fn)
    mpd_loss_and_grad_fn = nn.value_and_grad(mpd, mpd_loss_fn)
    g_loss_and_grad_fn = nn.value_and_grad(ae, g_loss_fn)
    total_steps = steps
    loader = dataloader(
        dataset,
        batch_size=batch_size,
        segment_seconds=segment_seconds,
        sample_rate=sample_rate,
        mel_config=mel_cfg,
        log_mel=True,
        seed=seed,
    )
    logger = MetricLogger()

    config_rows = [
        ("sr", sample_rate),
        ("segment_seconds", f"{segment_seconds:.5f}"),
        ("mel_nfft", mel_cfg.n_fft),
        ("mel_hop", mel_cfg.hop_length),
        ("mel_nmels", mel_cfg.n_mels),
        ("lr", learning_rate),
        ("disc_lr", disc_learning_rate),
        ("lr_schedule", lr_schedule),
        ("min_lr", min_learning_rate),
        ("warmup_steps", warmup_steps),
        ("steps", steps),
        ("seed", seed),
    ]
    print("config:\n" + format_table(config_rows, ["param", "value"]))

    progress = tqdm(range(steps), total=steps, desc="train")
    last_lr = learning_rate
    for step in progress:
        lr = compute_learning_rate(
            step,
            base_lr=learning_rate,
            min_lr=min_learning_rate,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            schedule=lr_schedule,
        )
        last_lr = lr
        set_optim_lr(opt_g, lr)
        set_optim_lr(opt_mrd, lr if disc_learning_rate == learning_rate else disc_learning_rate)
        set_optim_lr(opt_mpd, lr if disc_learning_rate == learning_rate else disc_learning_rate)

        batch = next(loader)
        mel = batch["mel"]
        real_waveform = batch["waveform"]
        fake_waveform = ae(mel)

        mrd_d_loss, mrd_grads = mrd_loss_and_grad_fn(mrd, real_waveform, fake_waveform)
        mpd_d_loss, mpd_grads = mpd_loss_and_grad_fn(mpd, real_waveform, fake_waveform)
        (g_loss_value, loss_dict), g_grads = g_loss_and_grad_fn(ae, mrd, mpd, mel, real_waveform)

        metrics = {
            "mrd_d": float(mrd_d_loss.item()),
            "mpd_d": float(mpd_d_loss.item()),
            "g": float(g_loss_value.item()),
            "reconstruction_loss": loss_dict["reconstruction_loss"],
            "adversarial_loss": loss_dict["adversarial_loss"],
            "feature_matching_loss": loss_dict["feature_matching_loss"],
        }
        logger.update(metrics)

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

        mrd_d_val = metrics["mrd_d"]
        mpd_d_val = metrics["mpd_d"]
        g_val = metrics["g"]

        progress.set_postfix(
            mrd_d=f"{mrd_d_val:.4f}",
            mpd_d=f"{mpd_d_val:.4f}",
            g=f"{g_val:.4f}",
            lr=f"{lr:.3g}",
        )

        if (step == 0) or ((step + 1) % log_every == 0) or (step + 1 == steps):
            window_means = logger.window_mean()
            log_rows = [
                ("step", f"{step + 1}/{steps}"),
                ("avg_mrd_d", window_means.get("mrd_d", 0.0)),
                ("avg_mpd_d", window_means.get("mpd_d", 0.0)),
                ("avg_g", window_means.get("g", 0.0)),
                ("avg_reconstruction_loss", window_means.get("reconstruction_loss", 0.0)),
                ("avg_adversarial_loss", window_means.get("adversarial_loss", 0.0)),
                ("avg_feature_matching_loss", window_means.get("feature_matching_loss", 0.0)),
                ("lr", last_lr),
            ]
            print(format_table(log_rows, ["metric", "value"]))
            logger.reset_window()

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

    total_means = logger.total_mean()
    summary_rows = [
        ("steps", steps),
        ("avg_mrd_d", total_means.get("mrd_d", 0.0)),
        ("avg_mpd_d", total_means.get("mpd_d", 0.0)),
        ("avg_g", total_means.get("g", 0.0)),
        ("avg_reconstruction_loss", total_means.get("reconstruction_loss", 0.0)),
        ("avg_adversarial_loss", total_means.get("adversarial_loss", 0.0)),
        ("avg_feature_matching_loss", total_means.get("feature_matching_loss", 0.0)),
        ("lr", last_lr),
    ]
    print("summary:\n" + format_table(summary_rows, ["metric", "value"]))

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
    save_weights(out_dir=Path("output") / "final", ae=ae, mrd=mrd, mpd=mpd)


def main() -> None:
    args = parse_args()
    run_train(args)


if __name__ == "__main__":
    main()
