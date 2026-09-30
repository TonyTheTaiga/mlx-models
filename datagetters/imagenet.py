"""
Download ImageNet-1k (ILSVRC 2012 CLS-LOC) from Kaggle and unpack it into per-class folders.

Layout: <outdir>/{train,val}/<wnid>/<image>.JPEG, plus <outdir>/synsets.txt
(one "<wnid> <names>" line per class, in label-index order).

Requires Kaggle credentials (KAGGLE_USERNAME/KAGGLE_KEY or ~/.kaggle/kaggle.json) and that you
have accepted the competition rules at
https://www.kaggle.com/competitions/imagenet-object-localization-challenge/rules

The archive is ~155 GB and the extracted images take about the same again.

Usage:
    python datagetters/imagenet.py --outdir ./data/imagenet
"""

import argparse
import base64
import csv
import io
import json
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from urllib.error import HTTPError

from tqdm import tqdm

KAGGLE_URL = (
    "https://www.kaggle.com/api/v1/competitions/data/download-all/"
    "imagenet-object-localization-challenge"
)
CLS_LOC = "ILSVRC/Data/CLS-LOC/"


def _kaggle_credentials() -> tuple[str, str]:
    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")
    if username and key:
        return username, key

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        with open(kaggle_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("username") and data.get("key"):
            return data["username"], data["key"]

    raise SystemExit(
        "Kaggle credentials not found. Set KAGGLE_USERNAME/KAGGLE_KEY or create "
        "~/.kaggle/kaggle.json (Kaggle > Settings > API > Create New Token)."
    )


class _StripAuthRedirect(urllib.request.HTTPRedirectHandler):
    # Kaggle redirects to a signed storage URL that rejects our Basic auth header.
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is not None:
            new_req.remove_header("Authorization")
        return new_req


def download(destination: Path) -> None:
    """Stream the archive to `destination`, resuming a partial download if one exists."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0

    user, key = _kaggle_credentials()
    token = base64.b64encode(f"{user}:{key}".encode("utf-8")).decode("utf-8")
    headers = {"Authorization": f"Basic {token}"}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    opener = urllib.request.build_opener(_StripAuthRedirect)
    try:
        response = opener.open(urllib.request.Request(KAGGLE_URL, headers=headers))
    except HTTPError as exc:
        if exc.code == 403:
            raise SystemExit(
                "Kaggle returned 403: accept the competition rules at "
                "https://www.kaggle.com/competitions/imagenet-object-localization-challenge/rules"
            ) from exc
        if exc.code == 416:  # requested range starts at the end: already complete
            partial.rename(destination)
            return
        raise

    with response:
        if offset and response.status != 206:
            offset = 0  # server ignored the range request; start over
        total = int(response.headers.get("content-length", 0)) + offset
        with (
            open(partial, "ab" if offset else "wb") as fout,
            tqdm(total=total, initial=offset, unit="B", unit_scale=True, desc="imagenet") as pbar,
        ):
            while chunk := response.read(8 * 1024 * 1024):
                fout.write(chunk)
                pbar.update(len(chunk))

    partial.rename(destination)


def extract(archive_path: Path, outdir: Path) -> None:
    with zipfile.ZipFile(archive_path) as zf:
        with zf.open("LOC_synset_mapping.txt") as f:
            (outdir / "synsets.txt").write_bytes(f.read())

        # Val images sit in one flat folder; their class is the first token of PredictionString.
        with zf.open("LOC_val_solution.csv") as f:
            reader = csv.DictReader(io.TextIOWrapper(f, encoding="utf-8"))
            val_labels = {row["ImageId"]: row["PredictionString"].split()[0] for row in reader}

        for split in ("train", "val"):
            split_dir = outdir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)

            prefix = f"{CLS_LOC}{split}/"
            members = [
                info
                for info in zf.infolist()
                if info.filename.startswith(prefix) and info.filename.endswith(".JPEG")
            ]
            for info in tqdm(members, desc=f"extract {split}"):
                name = Path(info.filename).name
                wnid = (
                    Path(info.filename).parent.name if split == "train" else val_labels[name[:-5]]
                )
                target = split_dir / wnid / name
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grab ImageNet-1k from Kaggle.")
    parser.add_argument("--outdir", type=Path, default=Path("data/imagenet"))
    parser.add_argument(
        "--keep-archive", action="store_true", help="Keep the ~155 GB zip after extracting."
    )
    args = parser.parse_args()

    outdir: Path = args.outdir
    archive_path = outdir / "imagenet-object-localization-challenge.zip"

    if archive_path.exists():
        print(f"Found existing archive at {archive_path}, skipping download.")
    else:
        download(archive_path)

    extract(archive_path, outdir)
    if not args.keep_archive:
        archive_path.unlink()

    for split in ("train", "val"):
        count = sum(1 for _ in (outdir / split).glob("*/*.JPEG"))
        print(f"{split}: {count} images")


if __name__ == "__main__":
    main()
