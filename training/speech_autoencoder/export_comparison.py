"""Export audio comparison data for the HTML viewer.

Usage:
    python -m training.speech_autoencoder.export_comparison \
        --weights-dir output/step_5000 \
        --sample-index 1 \
        --out-dir comparison_output
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np

from networks.speech_autoencoder.model import SpeechAutoEncoder
from training.speech_autoencoder.dataset import SpsCorpusDataset
from training.speech_autoencoder.mels import MelSpectrogramConfig, MelSpectrogramEncoder
from training.speech_autoencoder.utils import (
    reconstruct_mel_in_chunks,
    write_wav_mono,
)


def export_mel_image(
    mel: np.ndarray,
    out_path: Path,
    title: str = "",
    figsize: tuple[int, int] = (16, 4),
) -> None:
    """Export mel spectrogram as a PNG image with no axes/margins."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(
        mel.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
    )
    ax.set_axis_off()
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(out_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export comparison data for HTML viewer")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "sps-corpus-1.0-2025-11-25-en",
    )
    parser.add_argument(
        "--weights-dir",
        type=Path,
        required=True,
        help="Directory containing ae_weights.npz",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--name", type=str, default=None, help="Display name for this comparison")
    parser.add_argument("--sample-rate", type=int, default=32_000)
    parser.add_argument("--out-dir", type=Path, default=Path("comparison_output"))
    parser.add_argument("--chunk-frames", type=int, default=256)
    parser.add_argument("--overlap-frames", type=int, default=128)
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Mel config (must match training)
    mel_cfg = MelSpectrogramConfig(
        sample_rate=args.sample_rate,
        n_fft=1536,
        hop_length=256,
        win_length=1536,
        n_mels=228,
    )

    # Load model
    ae = SpeechAutoEncoder(in_dims=mel_cfg.n_mels, hidden_dims=24, out_dims=mel_cfg.hop_length)
    weights_path = args.weights_dir / "ae_weights.npz"
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    ae.load_weights(str(weights_path))
    mx.eval(ae.parameters())
    ae.train(False)

    # Load dataset and sample
    dataset = SpsCorpusDataset(dataset_dir=args.dataset_dir, split="train")
    sample = dataset[args.sample_index]
    wav_real, _ = sample.load_waveform(target_sr=args.sample_rate, as_mx=False)
    wav_real = np.asarray(wav_real, dtype=np.float32)
    orig_len = len(wav_real)

    # Align to hop length
    hop = mel_cfg.hop_length
    aligned_len = int(math.ceil(orig_len / hop) * hop)
    if aligned_len != orig_len:
        wav_padded = np.pad(wav_real, (0, aligned_len - orig_len), mode="constant")
    else:
        wav_padded = wav_real

    # Encode to mel
    encoder = MelSpectrogramEncoder(mel_cfg)
    mel_real = encoder.encode(wav_padded, log_mel=True, pad_mode="reflect", as_mx=False)
    mel_mx = mx.array(mel_real[None, :, :], dtype=mx.float32)

    # Reconstruct
    wav_fake_mx = reconstruct_mel_in_chunks(
        ae,
        mel_mx,
        out_dims=hop,
        chunk_frames=args.chunk_frames,
        overlap_frames=args.overlap_frames,
    )
    mx.eval(wav_fake_mx)
    wav_fake = np.asarray(wav_fake_mx)[0, :orig_len, 0]

    # Compute mel of reconstructed audio
    mel_fake = encoder.encode(wav_fake, log_mel=True, pad_mode="reflect", as_mx=False)

    # Save audio files
    write_wav_mono(out_dir / "real.wav", wav_real, args.sample_rate)
    write_wav_mono(out_dir / "fake.wav", wav_fake, args.sample_rate)

    # Save mel images
    export_mel_image(mel_real, out_dir / "mel_real.png", title="Original")
    export_mel_image(mel_fake, out_dir / "mel_fake.png", title="Reconstructed")

    # Save metadata
    duration = orig_len / args.sample_rate
    metadata = {
        "name": args.name,
        "sample_id": sample.audio_id,
        "sample_rate": args.sample_rate,
        "duration": duration,
        "num_frames": mel_real.shape[0],
        "num_mels": mel_real.shape[1],
        "hop_length": hop,
        "files": {
            "real_audio": "real.wav",
            "fake_audio": "fake.wav",
            "real_mel": "mel_real.png",
            "fake_mel": "mel_fake.png",
        },
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Exported comparison to {out_dir}")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Frames: {mel_real.shape[0]}")
    print(f"  Open viewer.html and load the metadata.json file")


if __name__ == "__main__":
    main()
