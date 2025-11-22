"""
Download and unpack WikiText-103 for masked language model pretraining.

Usage:
    python datagetters/wikitext_mlm.py --outdir ./data/wikitext-103
"""

import argparse
import base64
import csv
import json
import os
import urllib.request
from urllib.error import HTTPError
import zipfile
from pathlib import Path

from tqdm import tqdm


# Primary source: Kaggle dataset API.
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/vadimkurochkin/wikitext-103"
# Fallback: Hugging Face mirror.
WIKITEXT_URL_FALLBACK = "https://huggingface.co/datasets/wikitext/raw/main/wikitext-103-v1.zip"


def _kaggle_credentials() -> tuple[str, str] | None:
    # Prefer env vars, otherwise read ~/.kaggle/kaggle.json
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if username and key:
        return username, key

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        with open(kaggle_json, "r", encoding="utf-8") as f:
            data = json.load(f)
            username = data.get("username")
            key = data.get("key")
            if username and key:
                return username, key
    return None


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)

    def _stream(src_url: str, headers: dict[str, str] | None = None):
        req = urllib.request.Request(src_url, headers=headers or {})
        with urllib.request.urlopen(req) as response:
            total = int(response.headers.get("content-length", 0))
            with open(destination, "wb") as fout, tqdm(
                total=total, unit="B", unit_scale=True, desc=f"Downloading {destination.name}"
            ) as pbar:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    fout.write(chunk)
                    pbar.update(len(chunk))

    # Try Kaggle with auth first if available.
    creds = _kaggle_credentials() if url == KAGGLE_URL else None
    if creds:
        user, key = creds
        token = base64.b64encode(f"{user}:{key}".encode("utf-8")).decode("utf-8")
        headers = {"Authorization": f"Basic {token}"}
    else:
        headers = None

    try:
        _stream(url, headers=headers)
    except HTTPError as exc:
        if exc.code in {301, 302, 303, 307, 308}:
            redirect_url = exc.headers.get("Location")
            if redirect_url:
                print(f"Redirected ({exc.code}); retrying {redirect_url}")
                _stream(redirect_url, headers=headers)
                return destination
        print(f"Download failed with HTTP {exc.code}, trying fallback: {WIKITEXT_URL_FALLBACK}")
        _stream(WIKITEXT_URL_FALLBACK)

    return destination


def extract_zip(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(path=target_dir)


def _load_documents(tokens_path: Path) -> list[tuple[str, str]]:
    """Convert a .tokens file into text documents separated by blank lines."""
    docs: list[str] = []
    current: list[str] = []
    with open(tokens_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                if current:
                    if len(current) > 1:
                        docs.append(" ".join(current[1:]))  # drop title line
                    current = []
                continue
            current.append(stripped)
    if current:
        if len(current) > 1:
            docs.append(" ".join(current[1:]))  # drop title line
    return docs


def emit_csvs(extracted_root: Path, outdir: Path) -> dict[str, Path]:
    split_paths = {
        "train": extracted_root / "wiki.train.tokens",
        "validation": extracted_root / "wiki.valid.tokens",
        "test": extracted_root / "wiki.test.tokens",
    }

    written: dict[str, Path] = {}
    for split, path in split_paths.items():
        csv_path = outdir / f"wikitext_{split}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["text"])
            for doc in _load_documents(path):
                writer.writerow([doc])
        written[split] = csv_path
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Grab WikiText-103 for MLM pretraining.")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "wikitext-103",
        help="Where to place the extracted dataset.",
    )
    args = parser.parse_args()

    outdir: Path = args.outdir
    archive_path = outdir / "wikitext-103-v1.zip"

    if archive_path.exists():
        print(f"Found existing archive at {archive_path}, skipping download.")
    else:
        print(f"Downloading WikiText-103 to {archive_path}")
        download(KAGGLE_URL, archive_path)

    extracted_root = outdir / "wikitext-103"
    if extracted_root.exists():
        print(f"Extracted data already present at {extracted_root}, skipping extraction.")
    else:
        print(f"Extracting to {outdir}")
        extract_zip(archive_path, outdir)
        print("Done. Files available under:", extracted_root)

    csv_paths = emit_csvs(extracted_root, outdir)
    for split, path in csv_paths.items():
        print(f"Wrote {split} CSV with documents: {path}")


if __name__ == "__main__":
    main()
