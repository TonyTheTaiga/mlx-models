from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence, overload

import numpy as np
import polars as pl
import soundfile as sf
import soxr

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = ROOT / "data" / "common_voice" / "cv-corpus-24.0-2025-12-05" / "en"


@dataclass(slots=True)
class CommonVoiceSample:
    client_id: str
    audio_id: int
    audio_file: str
    duration_ms: int
    sentence_id: str
    sentence: str
    up_votes: int
    down_votes: int
    age: str | None
    gender: str | None
    locale: str
    split: str

    @property
    def duration_seconds(self) -> float:
        return self.duration_ms / 1000.0

    def audio_path(self, dataset_dir: Path | None = None) -> Path:
        base_dir = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR
        return base_dir / "clips" / self.audio_file

    def load_waveform(
        self,
        dataset_dir: Path | None = None,
        target_sr: int = 16_000,
        as_mx: bool = False,
    ):
        audio_path = self.audio_path(dataset_dir)
        waveform, sr = sf.read(audio_path, dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if sr != target_sr:
            waveform = soxr.resample(waveform, sr, target_sr, quality="HQ")
        waveform = waveform.astype(np.float32)
        if as_mx:
            import mlx.core as mx
            waveform = mx.array(waveform, dtype=mx.float32)
        return waveform, target_sr


class CommonVoiceDataset(Sequence[CommonVoiceSample]):
    def __init__(
        self,
        dataset_dir: Path | str | None = None,
        split: str | Sequence[str] | None = None,
        limit: int | None = None,
        min_votes: int = 0,
    ) -> None:
        base_dir = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR
        if not base_dir.exists():
            raise FileNotFoundError(f"Dataset directory '{base_dir}' not found")
        self.dataset_dir = base_dir
        self._requested_splits = (
            {split.lower()} if isinstance(split, str)
            else ({s.lower() for s in split} if split else None)
        )
        self._limit = limit
        self._min_votes = min_votes
        self._samples = self._load_samples()

    def _load_durations(self) -> dict[str, int]:
        path = self.dataset_dir / "clip_durations.tsv"
        if not path.exists():
            return {}
        df = pl.read_csv(path, separator="\t")
        return {row["clip"]: int(row["duration[ms]"]) for row in df.iter_rows(named=True)}

    def _load_split(self, split: str, durations: dict[str, int]) -> list[CommonVoiceSample]:
        path = self.dataset_dir / f"{split}.tsv"
        if not path.exists():
            return []
        df = pl.read_csv(
            path, separator="\t", null_values=[""], infer_schema_length=0,
            quote_char=None, ignore_errors=True,
        )
        samples = []
        for idx, row in enumerate(df.iter_rows(named=True)):
            audio_file = row["path"]
            up_votes = int(row.get("up_votes") or 0)
            down_votes = int(row.get("down_votes") or 0)
            if up_votes - down_votes < self._min_votes:
                continue
            samples.append(CommonVoiceSample(
                client_id=str(row["client_id"]),
                audio_id=idx,
                audio_file=audio_file,
                duration_ms=durations.get(audio_file, 0),
                sentence_id=str(row.get("sentence_id") or ""),
                sentence=str(row.get("sentence") or ""),
                up_votes=up_votes,
                down_votes=down_votes,
                age=row.get("age") or None,
                gender=row.get("gender") or None,
                locale=str(row.get("locale") or "en"),
                split=split,
            ))
            if self._limit and len(samples) >= self._limit:
                break
        return samples

    def _load_samples(self) -> tuple[CommonVoiceSample, ...]:
        durations = self._load_durations()
        splits = list(self._requested_splits) if self._requested_splits else ["train", "dev", "test"]
        samples = []
        for split in splits:
            samples.extend(self._load_split(split, durations))
            if self._limit and len(samples) >= self._limit:
                samples = samples[:self._limit]
                break
        return tuple(samples)

    def __len__(self) -> int:
        return len(self._samples)

    @overload
    def __getitem__(self, idx: int) -> CommonVoiceSample: ...

    @overload
    def __getitem__(self, idx: slice) -> tuple[CommonVoiceSample, ...]: ...

    def __getitem__(self, idx: int | slice) -> CommonVoiceSample | tuple[CommonVoiceSample, ...]:
        return self._samples[idx]

    def __iter__(self) -> Iterator[CommonVoiceSample]:
        return iter(self._samples)

    @property
    def total_duration_hours(self) -> float:
        return sum(s.duration_ms for s in self._samples) / 1000.0 / 3600.0

    def split_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for s in self._samples:
            dist[s.split] = dist.get(s.split, 0) + 1
        return dist


if __name__ == "__main__":
    dataset = CommonVoiceDataset(split="train", limit=5)
    print(f"Loaded {len(dataset)} samples ({dataset.total_duration_hours:.2f}h)")
    print("Split distribution:", dataset.split_distribution())
    for idx, sample in enumerate(dataset):
        print(f"[{idx}] {sample.audio_file} ({sample.duration_seconds:.2f}s)")
        print(f"     {sample.sentence}")
