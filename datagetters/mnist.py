"""
Download MNIST and unpack it into per-label PNG folders.

Layout: <outdir>/{train,test}/<label>/<index>.png

Usage:
    python datagetters/mnist.py --outdir ./data/mnist
"""

import argparse
import gzip
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

# yann.lecun.com frequently rejects requests; this is the mirror torchvision uses.
MNIST_URL = "https://ossci-datasets.s3.amazonaws.com/mnist"
SPLITS = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def _download(name: str, cache_dir: Path) -> Path:
    path = cache_dir / name
    if not path.exists():
        print(f"Downloading {name}")
        # Write to a temp name so an interrupted download isn't mistaken for a complete one.
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(f"{MNIST_URL}/{name}", partial)
        partial.rename(path)
    return path


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]
    shape = tuple(int.from_bytes(data[4 + 4 * i : 8 + 4 * i], "big") for i in range(ndim))
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("data/mnist"))
    args = parser.parse_args()

    cache_dir = args.outdir / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for split, (images_name, labels_name) in SPLITS.items():
        images = _read_idx(_download(images_name, cache_dir))
        labels = _read_idx(_download(labels_name, cache_dir))
        assert len(images) == len(labels)

        split_dir = args.outdir / split
        if split_dir.exists():
            shutil.rmtree(split_dir)
        for label in range(10):
            (split_dir / str(label)).mkdir(parents=True)

        for index, (image, label) in enumerate(
            tqdm(zip(images, labels), total=len(images), desc=split)
        ):
            cv2.imwrite(str(split_dir / str(label) / f"{index:05d}.png"), image)


if __name__ == "__main__":
    main()
