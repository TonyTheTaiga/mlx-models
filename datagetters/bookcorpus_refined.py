"""
Download the refined BookCorpus dataset using the Kaggle datasets HTTP endpoint (no kagglehub dependency).

Requires Kaggle credentials (KAGGLE_USERNAME / KAGGLE_KEY env vars or ~/.kaggle/kaggle.json).

Usage:
    python datagetters/bookcorpus_refined.py --outdir ./data/bookcorpus-refined
"""

import argparse
import base64
import json
import os
import urllib.request
from urllib.error import HTTPError
import zipfile
from pathlib import Path

from tqdm import tqdm


DATASET_URL = "https://www.kaggle.com/api/v1/datasets/download/nishantsingh96/refined-bookcorpus-dataset"
DEFAULT_ARCHIVE_NAME = "bookcorpus-refined.zip"


def _kaggle_credentials() -> tuple[str, str] | None:
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


def download(url: str, destination: Path, auth: tuple[str, str]) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    user, key = auth
    token = base64.b64encode(f"{user}:{key}".encode("utf-8")).decode("utf-8")
    headers = {"Authorization": f"Basic {token}"}

    def _stream(src_url: str):
        req = urllib.request.Request(src_url, headers=headers)
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

    try:
        _stream(url)
    except HTTPError as exc:
        if exc.code in {301, 302, 303, 307, 308}:
            redirect_url = exc.headers.get("Location")
            if redirect_url:
                print(f"Redirected ({exc.code}); retrying {redirect_url}")
                _stream(redirect_url)
                return destination
        raise

    return destination


def extract_zip(archive_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(path=target_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grab the refined BookCorpus dataset via Kaggle HTTP API."
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "bookcorpus-refined",
        help="Where to place the dataset locally.",
    )
    parser.add_argument(
        "--archive-name",
        type=str,
        default=DEFAULT_ARCHIVE_NAME,
        help="Filename to use for the downloaded zip archive.",
    )
    args = parser.parse_args()

    outdir: Path = args.outdir
    archive_path = outdir / args.archive_name
    extracted_dir = outdir / "extracted"

    creds = _kaggle_credentials()
    if not creds:
        raise RuntimeError(
            "Kaggle credentials not found. Set KAGGLE_USERNAME and KAGGLE_KEY or create ~/.kaggle/kaggle.json"
        )

    if archive_path.exists():
        print(f"Found existing archive at {archive_path}, skipping download.")
    else:
        print(f"Downloading refined BookCorpus to {archive_path}")
        download(DATASET_URL, archive_path, auth=creds)

    if extracted_dir.exists():
        print(f"Extracted data already present at {extracted_dir}, skipping extraction.")
    else:
        print(f"Extracting to {extracted_dir}")
        extract_zip(archive_path, extracted_dir)
        print(f"Done. Files available under: {extracted_dir}")


if __name__ == "__main__":
    main()
