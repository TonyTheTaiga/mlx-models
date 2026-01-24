import argparse
import math
import random
from pathlib import Path
from typing import Iterator

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tabulate import tabulate as _tabulate
from tqdm import tqdm

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


def save_weights(
    out_dir: Path,
    ae: SpeechAutoEncoder,
    mrd: MRD,
    mpd: MPD,
    opt_g: optim.Optimizer,
    opt_mrd: optim.Optimizer,
    opt_mpd: optim.Optimizer,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.eval(ae.parameters(), mrd.parameters(), mpd.parameters())
    ae.save_weights(str(out_dir / "ae_weights.npz"))
    mrd.save_weights(str(out_dir / "mrd_weights.npz"))
    mpd.save_weights(str(out_dir / "mpd_weights.npz"))
    _save_optimizer_state(out_dir / "opt_g_state.npz", opt_g)
    _save_optimizer_state(out_dir / "opt_mrd_state.npz", opt_mrd)
    _save_optimizer_state(out_dir / "opt_mpd_state.npz", opt_mpd)


def _save_optimizer_state(path: Path, opt: optim.Optimizer) -> None:
    from mlx.utils import tree_flatten

    flat_state = tree_flatten(opt.state)
    if not flat_state:
        return
    arrays = {f"s{i}": v for i, (_, v) in enumerate(flat_state)}
    mx.savez(str(path), **arrays)


def _load_optimizer_state(path: Path, opt: optim.Optimizer) -> bool:
    from mlx.utils import tree_flatten, tree_unflatten

    if not path.exists():
        return False
    loaded = dict(mx.load(str(path)))
    if not loaded:
        return False
    flat_state = tree_flatten(opt.state)
    if len(flat_state) != len(loaded):
        print(f"Warning: optimizer state size mismatch at {path}, skipping load")
        return False
    new_flat = [(k, loaded[f"s{i}"]) for i, (k, _) in enumerate(flat_state)]
    opt.state = tree_unflatten(new_flat)
    return True


def _init_optimizer_state(opt: optim.Optimizer, model: nn.Module) -> None:
    from mlx.utils import tree_map

    params = model.parameters()
    original = tree_map(lambda p: mx.array(p), params)
    zero_grads = tree_map(lambda p: mx.zeros_like(p), params)
    opt.update(model, zero_grads)
    model.update(original)
    mx.eval(model.parameters(), opt.state)


def find_latest_checkpoint(output_dir: Path) -> tuple[Path | None, int]:
    if not output_dir.exists():
        return None, 0

    checkpoint_dirs = []
    for path in output_dir.iterdir():
        if path.is_dir() and path.name.startswith("step_"):
            try:
                step = int(path.name.split("_")[1])
                checkpoint_dirs.append((path, step))
            except (ValueError, IndexError):
                continue

    if not checkpoint_dirs:
        return None, 0

    checkpoint_dirs.sort(key=lambda x: x[1])
    return checkpoint_dirs[-1]


def load_checkpoint(
    checkpoint_dir: Path,
    ae: SpeechAutoEncoder,
    mrd: MRD,
    mpd: MPD,
    opt_g: optim.Optimizer,
    opt_mrd: optim.Optimizer,
    opt_mpd: optim.Optimizer,
) -> None:
    ae_weights_path = checkpoint_dir / "ae_weights.npz"
    mrd_weights_path = checkpoint_dir / "mrd_weights.npz"
    mpd_weights_path = checkpoint_dir / "mpd_weights.npz"

    if not all([ae_weights_path.exists(), mrd_weights_path.exists(), mpd_weights_path.exists()]):
        raise FileNotFoundError(f"Missing weight files in {checkpoint_dir}")

    ae.load_weights(str(ae_weights_path))
    mrd.load_weights(str(mrd_weights_path))
    mpd.load_weights(str(mpd_weights_path))
    mx.eval(ae.parameters(), mrd.parameters(), mpd.parameters())

    _init_optimizer_state(opt_g, ae)
    _init_optimizer_state(opt_mrd, mrd)
    _init_optimizer_state(opt_mpd, mpd)

    opt_g_loaded = _load_optimizer_state(checkpoint_dir / "opt_g_state.npz", opt_g)
    opt_mrd_loaded = _load_optimizer_state(checkpoint_dir / "opt_mrd_state.npz", opt_mrd)
    opt_mpd_loaded = _load_optimizer_state(checkpoint_dir / "opt_mpd_state.npz", opt_mpd)

    if not all([opt_g_loaded, opt_mrd_loaded, opt_mpd_loaded]):
        print("Warning: Some optimizer states not found, using fresh optimizer state")

    mx.eval(opt_g.state, opt_mrd.state, opt_mpd.state)


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
        self.total_counts: dict[str, int] = {}
        self.window_counts: dict[str, int] = {}
        self.last: dict[str, float] = {}

    def update(self, metrics: dict[str, float]) -> None:
        metrics = {k: float(v) for k, v in metrics.items()}
        self.last = metrics
        for key, value in metrics.items():
            self.total[key] = self.total.get(key, 0.0) + value
            self.window[key] = self.window.get(key, 0.0) + value
            self.total_counts[key] = self.total_counts.get(key, 0) + 1
            self.window_counts[key] = self.window_counts.get(key, 0) + 1

    def window_mean(self) -> dict[str, float]:
        return {k: v / max(self.window_counts.get(k, 0), 1) for k, v in self.window.items()}

    def total_mean(self) -> dict[str, float]:
        return {k: v / max(self.total_counts.get(k, 0), 1) for k, v in self.total.items()}

    def reset_window(self) -> None:
        self.window.clear()
        self.window_counts.clear()


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
    parser.add_argument("--batch-size", type=int, default=16)
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
        default=1e-6,
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
        hop_length=256,
        win_length=1536,
        n_mels=228,
    )

    ae = SpeechAutoEncoder(in_dims=mel_cfg.n_mels, hidden_dims=24, out_dims=mel_cfg.hop_length)
    mrd = MRD(fft_sizes=[384, 768, 1536])
    mpd = MPD()

    opt_g = optim.AdamW(learning_rate=learning_rate)
    opt_mrd = optim.AdamW(learning_rate=disc_learning_rate)
    opt_mpd = optim.AdamW(learning_rate=disc_learning_rate)
    mx.eval(ae.parameters(), mrd.parameters(), mpd.parameters())

    output_dir = Path("output")
    checkpoint_path, start_step = find_latest_checkpoint(output_dir)
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path} (step {start_step})")
        load_checkpoint(checkpoint_path, ae, mrd, mpd, opt_g, opt_mrd, opt_mpd)
    else:
        start_step = 0
        print("No checkpoint found, starting from scratch")
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
        ("start_step", start_step),
        ("seed", seed),
    ]
    print("config:\n" + format_table(config_rows, ["param", "value"]))

    progress = tqdm(range(start_step, steps), total=steps, initial=start_step, desc="train")
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

        batch = next(loader)
        mel = batch["mel"]
        real_waveform = batch["waveform"]

        fake_waveform = ae(mel)
        fake_waveform_detached = mx.stop_gradient(fake_waveform)
        mrd_d_loss, mrd_grads = mrd_loss_and_grad_fn(mrd, real_waveform, fake_waveform_detached)
        mpd_d_loss, mpd_grads = mpd_loss_and_grad_fn(mpd, real_waveform, fake_waveform_detached)
        opt_mrd.update(mrd, mrd_grads)
        opt_mpd.update(mpd, mpd_grads)

        (g_loss_value, loss_dict), g_grads = g_loss_and_grad_fn(ae, mrd, mpd, mel, real_waveform)
        opt_g.update(ae, g_grads)
        mx.eval(
            ae.parameters(),
            mrd.parameters(),
            mpd.parameters(),
            opt_g.state,
            opt_mrd.state,
            opt_mpd.state,
        )

        metrics = {
            "g": float(g_loss_value.item()),
            "reconstruction_loss": float(loss_dict["reconstruction_loss"]),
            "adversarial_loss": float(loss_dict["adversarial_loss"]),
            "feature_matching_loss": float(loss_dict["feature_matching_loss"]),
            "mrd_d": float(mrd_d_loss.item()),
            "mpd_d": float(mpd_d_loss.item()),
        }

        logger.update(metrics)

        progress.set_postfix(
            mrd_d=f"{metrics['mrd_d']:.4f}",
            mpd_d=f"{metrics['mpd_d']:.4f}",
            g=f"{metrics['g']:.4f}",
            lr=f"{lr:.3g}",
        )

        if (step == 0) or ((step + 1) % log_every == 0) or (step + 1 == steps):
            window_means = logger.window_mean()
            log_rows = [
                ("step", f"{step + 1}/{steps}"),
                ("avg_mrd_d", window_means.get("mrd_d", 0.0)),
                ("avg_mpd_d", window_means.get("mpd_d", 0.0)),
                ("avg_g", window_means.get("g", float("nan"))),
                ("avg_reconstruction_loss", window_means.get("reconstruction_loss", float("nan"))),
                ("avg_adversarial_loss", window_means.get("adversarial_loss", float("nan"))),
                (
                    "avg_feature_matching_loss",
                    window_means.get("feature_matching_loss", float("nan")),
                ),
                ("lr", last_lr),
            ]
            print(format_table(log_rows, ["metric", "value"]))
            logger.reset_window()

        if save_every > 0 and (step + 1) % save_every == 0:
            out_dir = Path("output") / f"step_{step + 1}"
            random_sample_idx = random.randint(0, len(dataset) - 1)
            save_full_reconstruction(
                dataset=dataset,
                model=ae,
                mel_cfg=mel_cfg,
                sample_rate=sample_rate,
                out_dir=out_dir,
                sample_index=random_sample_idx,
                chunk_frames=256,
                overlap_frames=128,
            )
            save_weights(
                out_dir=out_dir,
                ae=ae,
                mrd=mrd,
                mpd=mpd,
                opt_g=opt_g,
                opt_mrd=opt_mrd,
                opt_mpd=opt_mpd,
            )

    total_means = logger.total_mean()
    summary_rows = [
        ("steps", steps),
        ("avg_mrd_d", total_means.get("mrd_d", 0.0)),
        ("avg_mpd_d", total_means.get("mpd_d", 0.0)),
        ("avg_g", total_means.get("g", float("nan"))),
        ("avg_reconstruction_loss", total_means.get("reconstruction_loss", float("nan"))),
        ("avg_adversarial_loss", total_means.get("adversarial_loss", float("nan"))),
        ("avg_feature_matching_loss", total_means.get("feature_matching_loss", float("nan"))),
        ("lr", last_lr),
    ]
    print("summary:\n" + format_table(summary_rows, ["metric", "value"]))

    random_sample_idx = random.randint(0, len(dataset) - 1)
    save_full_reconstruction(
        dataset=dataset,
        model=ae,
        mel_cfg=mel_cfg,
        sample_rate=sample_rate,
        out_dir=Path("output") / "final",
        sample_index=random_sample_idx,
        chunk_frames=256,
        overlap_frames=128,
    )
    save_weights(
        out_dir=Path("output") / "final",
        ae=ae,
        mrd=mrd,
        mpd=mpd,
        opt_g=opt_g,
        opt_mrd=opt_mrd,
        opt_mpd=opt_mpd,
    )


def main() -> None:
    args = parse_args()
    run_train(args)


if __name__ == "__main__":
    main()
