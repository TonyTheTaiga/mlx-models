import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence, overload

import numpy as np
import polars as pl
import soundfile as sf
import soxr

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = ROOT / "data" / "sps-corpus-1.0-2025-11-25-en"
METADATA_FILENAME = "ss-corpus-en.tsv"
REPORTED_FILENAME = "ss-reported-audios-en.tsv"

QUALITY_FLAG_TO_FIELD = {
    "transcription-length": "transcription_length",
    "speech-rate": "speech_rate",
    "short-audio": "short_audio",
    "long-audio": "long_audio",
    "non-allowed-script": "non_allowed_script",
    "mixed-script-words": "mixed_script_words",
    "mixed-script-transcription": "mixed_script_transcription",
}


def ensure_dataset_dir(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset directory '{path}' was not found. "
            "Make sure you have downloaded sps-corpus-1.0-2025-11-25-en "
            "into the repo's data/ directory."
        )
    return path


def normalize_optional(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def parse_quality_tags(raw_tags: str | None) -> tuple[tuple[str, ...], dict[str, bool]]:
    tags = tuple(tag for tag in (raw_tags or "").split("|") if tag)
    flag_values = {field: False for field in QUALITY_FLAG_TO_FIELD.values()}
    for tag in tags:
        target_field = QUALITY_FLAG_TO_FIELD.get(tag)
        if target_field:
            flag_values[target_field] = True
    return tags, flag_values


def decode_audio(audio_path: Path, target_sr: int) -> tuple[np.ndarray, int]:
    waveform, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    if sr != target_sr:
        waveform = soxr.resample(waveform, sr, target_sr, quality="HQ")
    return waveform.astype(np.float32), target_sr


def decode_audio_file(audio_path: Path | str, target_sr: int = 16_000) -> tuple[np.ndarray, int]:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file '{path}' was not found")
    return decode_audio(path, target_sr)


@dataclass(slots=True, frozen=True)
class ReportedIssue:
    reason: str
    comment: str | None


@dataclass(slots=True)
class SpeechSample:
    client_id: str
    audio_id: int
    audio_file: str
    duration_ms: int
    prompt_id: int
    prompt: str
    transcription: str
    votes: int
    age: str | None
    gender: str | None
    language: str
    split: str
    char_per_sec: float | None
    quality_tags: tuple[str, ...]
    transcription_length: bool
    speech_rate: bool
    short_audio: bool
    long_audio: bool
    non_allowed_script: bool
    mixed_script_words: bool
    mixed_script_transcription: bool
    reported_reason: str | None = None
    reported_comment: str | None = None

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0

    def audio_path(self, dataset_dir: Path | None = None) -> Path:
        base_dir = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR
        return base_dir / "audios" / self.audio_file

    def load_waveform(
        self,
        dataset_dir: Path | None = None,
        target_sr: int = 16_000,
        as_mx: bool = False,
    ):
        audio_path = self.audio_path(dataset_dir)
        waveform, sample_rate = decode_audio_file(audio_path, target_sr)
        if as_mx:
            import mlx.core as mx

            waveform = mx.array(waveform, dtype=mx.float32)
        return waveform, sample_rate


def load_reported_map(dataset_dir: Path) -> dict[int, ReportedIssue]:
    report_path = dataset_dir / REPORTED_FILENAME
    if not report_path.exists():
        return {}

    df = pl.read_csv(report_path, separator="\t", null_values=[""], infer_schema_length=0)
    reported: dict[int, ReportedIssue] = {}
    for row in df.iter_rows(named=True):
        audio_id = int(row["audio_id"])
        reported[audio_id] = ReportedIssue(
            reason=row.get("reason") or "unspecified",
            comment=normalize_optional(row.get("comment")),
        )
    return reported


def load_metadata_frame(dataset_dir: Path) -> pl.DataFrame:
    metadata_path = dataset_dir / METADATA_FILENAME
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file '{metadata_path}' not found. Please download the TSV first."
        )

    df = pl.read_csv(metadata_path, separator="\t", null_values=[""], infer_schema_length=0)
    return df


def row_to_sample(
    row: dict[str, object],
    reported_map: dict[int, ReportedIssue],
) -> SpeechSample:
    tags, flag_values = parse_quality_tags(row.get("quality_tags"))
    audio_id = int(row["audio_id"])
    reported_issue = reported_map.get(audio_id)
    return SpeechSample(
        client_id=str(row["client_id"]),
        audio_id=audio_id,
        audio_file=str(row["audio_file"]),
        duration_ms=int(row["duration_ms"]),
        prompt_id=int(row["prompt_id"]),
        prompt=str(row.get("prompt") or ""),
        transcription=str(row.get("transcription") or ""),
        votes=int(row["votes"] or 0),
        age=normalize_optional(row.get("age")),
        gender=normalize_optional(row.get("gender")),
        language=str(row.get("language") or "unknown"),
        split=str(row.get("split") or "unknown"),
        char_per_sec=float(row["char_per_sec"]) if row.get("char_per_sec") is not None else None,
        quality_tags=tags,
        transcription_length=flag_values["transcription_length"],
        speech_rate=flag_values["speech_rate"],
        short_audio=flag_values["short_audio"],
        long_audio=flag_values["long_audio"],
        non_allowed_script=flag_values["non_allowed_script"],
        mixed_script_words=flag_values["mixed_script_words"],
        mixed_script_transcription=flag_values["mixed_script_transcription"],
        reported_reason=reported_issue.reason if reported_issue else None,
        reported_comment=reported_issue.comment if reported_issue else None,
    )


class SpsCorpusDataset(Sequence[SpeechSample]):
    def __init__(
        self,
        dataset_dir: Path | str | None = None,
        split: str | Sequence[str] | None = None,
        limit: int | None = None,
        drop_quality_tags: Sequence[str] | None = None,
        exclude_reported: bool = False,
    ) -> None:
        base_dir = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR
        self.dataset_dir = ensure_dataset_dir(base_dir)
        self._requested_splits = (
            {split.lower()}
            if isinstance(split, str)
            else ({s.lower() for s in split} if split else None)
        )
        self._limit = limit
        self._drop_quality_tags = set(drop_quality_tags or [])
        self._exclude_reported = exclude_reported
        self._samples = self._load_samples()

    def _load_samples(self) -> tuple[SpeechSample, ...]:
        df = load_metadata_frame(self.dataset_dir)
        reported_map = load_reported_map(self.dataset_dir)

        if self._requested_splits:
            df = df.filter(pl.col("split").str.to_lowercase().is_in(list(self._requested_splits)))

        samples: list[SpeechSample] = []
        for row in df.iter_rows(named=True):
            sample = row_to_sample(row, reported_map)
            if self._drop_quality_tags and any(
                tag in self._drop_quality_tags for tag in sample.quality_tags
            ):
                continue
            if self._exclude_reported and sample.reported_reason is not None:
                continue
            samples.append(sample)
            if self._limit is not None and len(samples) >= self._limit:
                break
        return tuple(samples)

    def __len__(self) -> int:
        return len(self._samples)

    @overload
    def __getitem__(self, idx: int) -> SpeechSample: ...

    @overload
    def __getitem__(self, idx: slice) -> tuple[SpeechSample, ...]: ...

    def __getitem__(self, idx: int | slice) -> SpeechSample | tuple[SpeechSample, ...]:
        return self._samples[idx]

    def __iter__(self) -> Iterator[SpeechSample]:
        return iter(self._samples)

    @property
    def total_duration_hours(self) -> float:
        total_ms = sum(sample.duration_ms for sample in self._samples)
        return total_ms / 1000.0 / 3600.0

    def split_distribution(self) -> dict[str, int]:
        distribution: dict[str, int] = {}
        for sample in self._samples:
            key = sample.split
            distribution[key] = distribution.get(key, 0) + 1
        return distribution


def main(args: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect the SPS corpus metadata.")
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Filter to a single split (train/dev/test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of examples to preview.",
    )
    parser.add_argument(
        "--drop-quality-tags",
        type=str,
        nargs="*",
        default=None,
        help="Drop samples that contain any of the provided quality tags.",
    )
    parser.add_argument(
        "--exclude-reported",
        action="store_true",
        help="Exclude clips that were reported by annotators.",
    )
    parsed = parser.parse_args(args=args)

    dataset = SpsCorpusDataset(
        split=parsed.split,
        limit=parsed.limit,
        drop_quality_tags=parsed.drop_quality_tags,
        exclude_reported=parsed.exclude_reported,
    )

    print(
        f"Loaded {len(dataset)} samples "
        f"({dataset.total_duration_hours:.2f}h) "
        f"from {dataset.dataset_dir}"
    )
    print("Split distribution:", dataset.split_distribution())

    for idx, sample in enumerate(dataset):
        print(
            f"[{idx}] audio={sample.audio_file} "
            f"duration={sample.duration_seconds:.2f}s "
            f"quality_tags={sample.quality_tags}"
        )
        print(f"     prompt: {sample.prompt}")
        print(f"     transcription: {sample.transcription}")


if __name__ == "__main__":
    main()
