"""
Download the DIV2K high-resolution images (Agustsson & Timofte, NTIRE 2017) for super resolution.

Layout: <outdir>/{train,valid}/<index>.png — 800 train and 100 valid 2K images. Low-resolution
inputs are made on the fly by the trainer, so only the HR sets are fetched (~3.9 GB).

Usage:
    python datagetters/div2k.py --outdir ./data/div2k
"""

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

DIV2K_URL = "http://data.vision.ee.ethz.ch/cvl/DIV2K"
SPLITS = {"train": "DIV2K_train_HR.zip", "valid": "DIV2K_valid_HR.zip"}


def _download(name: str, cache_dir: Path) -> Path:
    path = cache_dir / name
    if not path.exists():
        # Write to a temp name so an interrupted download isn't mistaken for a complete one.
        partial = path.with_suffix(path.suffix + ".part")
        with (
            urllib.request.urlopen(f"{DIV2K_URL}/{name}") as response,
            open(partial, "wb") as fout,
            tqdm(
                total=int(response.headers.get("content-length", 0)),
                unit="B",
                unit_scale=True,
                desc=name,
            ) as pbar,
        ):
            while chunk := response.read(8 * 1024 * 1024):
                fout.write(chunk)
                pbar.update(len(chunk))
        partial.rename(path)
    return path


def _extract(archive_path: Path, split_dir: Path) -> None:
    if split_dir.exists():
        shutil.rmtree(split_dir)
    split_dir.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as zf:
        members = [info for info in zf.infolist() if info.filename.endswith(".png")]
        for info in tqdm(members, desc=f"extract {split_dir.name}"):
            with zf.open(info) as src, open(split_dir / Path(info.filename).name, "wb") as dst:
                shutil.copyfileobj(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser(description="Grab the DIV2K HR images.")
    parser.add_argument("--outdir", type=Path, default=Path("data/div2k"))
    parser.add_argument(
        "--keep-archive", action="store_true", help="Keep the zips after extracting."
    )
    args = parser.parse_args()

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    for split, name in SPLITS.items():
        archive_path = _download(name, outdir)
        _extract(archive_path, outdir / split)
        if not args.keep_archive:
            archive_path.unlink()
        count = sum(1 for _ in (outdir / split).glob("*.png"))
        print(f"{split}: {count} images")


if __name__ == "__main__":
    main()
