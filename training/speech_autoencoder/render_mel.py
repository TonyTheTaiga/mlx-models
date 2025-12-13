from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.speech_autoencoder.dataset import decode_audio_file
from training.speech_autoencoder.mels import MelSpectrogramDecoder, MelSpectrogramEncoder

DEFAULT_OUTPUT_DIR = ROOT / "training" / "speech_autoencoder" / "mel_images"
DEFAULT_AUDIO_DIR = ROOT / "training" / "speech_autoencoder" / "mel_recon_audio"
LOG_TO_DB = 10.0 / np.log(10.0)

COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
    "magma": cv2.COLORMAP_MAGMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "turbo": cv2.COLORMAP_TURBO,
    "cividis": cv2.COLORMAP_CIVIDIS,
    "hot": cv2.COLORMAP_HOT,
}


def normalize_mel_image(
    mel: np.ndarray,
    *,
    log_input: bool,
    db_range: float,
) -> np.ndarray:
    if mel.ndim != 2:
        raise ValueError("mel spectrogram must have shape (frames, bins)")

    if log_input:
        mel_db = mel * LOG_TO_DB
    else:
        mel_db = 10.0 * np.log10(np.maximum(mel, np.finfo(np.float32).eps))

    mel_db -= np.max(mel_db)
    mel_db = np.clip(mel_db, -db_range, 0.0)
    mel_norm = (mel_db + db_range) / db_range
    mel_norm = np.clip(mel_norm, 0.0, 1.0)

    mel_norm = np.flipud(mel_norm.T)
    mel_img = (mel_norm * 255.0).astype(np.uint8)
    return mel_img


def save_mel_image(
    audio_path: Path,
    output_dir: Path,
    encoder: MelSpectrogramEncoder,
    *,
    log_mel: bool,
    db_range: float,
    colormap: str,
    overwrite: bool,
) -> tuple[Path, np.ndarray]:
    waveform, _ = decode_audio_file(audio_path, encoder.config.sample_rate)
    mel = encoder.encode(waveform, log_mel=log_mel, as_mx=False)
    mel_img = normalize_mel_image(mel, log_input=log_mel, db_range=db_range)

    if colormap == "gray":
        colored = cv2.cvtColor(mel_img, cv2.COLOR_GRAY2BGR)
    else:
        cmap = COLORMAPS[colormap]
        colored = cv2.applyColorMap(mel_img, cmap)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{audio_path.stem}_mel.png"
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Use --overwrite to replace existing files."
        )
    cv2.imwrite(str(out_path), colored)
    return out_path, mel


def save_waveform(
    output_path: Path,
    waveform: np.ndarray,
    sample_rate: int,
    *,
    overwrite: bool,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Use --overwrite to replace existing files."
        )
    clipped = np.clip(waveform, -1.0, 1.0)
    int16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int16.tobytes())
    return output_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert audio files into mel spectrogram images.")
    parser.add_argument("audio", nargs="+", type=Path, help="Path(s) to audio files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory where mel images are saved (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use linear mel magnitudes instead of log mel scale.",
    )
    parser.add_argument(
        "--db-range",
        type=float,
        default=80.0,
        help="Dynamic range in decibels to show in the spectrogram image.",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="inferno",
        choices=("gray", *COLORMAPS.keys()),
        help="OpenCV colormap used for visualization.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing mel image files.",
    )
    parser.add_argument(
        "--reconstruct",
        action="store_true",
        help="Also reconstruct waveform from mel spectrograms and save as WAV files.",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_AUDIO_DIR,
        help=f"Directory to store reconstructed WAV files (default: {DEFAULT_AUDIO_DIR}).",
    )
    parser.add_argument(
        "--griffin-iters",
        type=int,
        default=32,
        help="Number of Griffin-Lim iterations when reconstructing audio.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    encoder = MelSpectrogramEncoder()
    decoder = None
    if args.reconstruct:
        decoder = MelSpectrogramDecoder(
            encoder=encoder,
            griffin_lim_iterations=max(args.griffin_iters, 1),
        )

    log_mel = not args.linear
    successes = []
    failures = []

    for audio_path in args.audio:
        try:
            img_path, mel = save_mel_image(
                audio_path=audio_path,
                output_dir=args.output_dir,
                encoder=encoder,
                log_mel=log_mel,
                db_range=args.db_range,
                colormap=args.colormap,
                overwrite=args.overwrite,
            )
            recon_path = None
            if decoder is not None:
                waveform = decoder.decode(mel, log_mel=log_mel, as_mx=False)
                recon_filename = f"{audio_path.stem}_recon.wav"
                recon_path = save_waveform(
                    args.audio_dir / recon_filename,
                    waveform,
                    encoder.config.sample_rate,
                    overwrite=args.overwrite,
                )
            successes.append((img_path, recon_path))
            msg = f"[OK] {audio_path} -> {img_path}"
            if recon_path:
                msg += f", {recon_path}"
            print(msg)
        except Exception as exc:  # noqa: BLE001
            failures.append((audio_path, exc))
            print(f"[ERR] {audio_path}: {exc}")

    print(f"Converted {len(successes)} file(s).")
    if failures:
        print(f"{len(failures)} file(s) failed:")
        for audio_path, exc in failures:
            print(f"  - {audio_path}: {exc}")


if __name__ == "__main__":
    main()
