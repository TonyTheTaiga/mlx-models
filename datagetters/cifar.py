"""
Download CIFAR-10 or CIFAR-100 and unpack it into per-label PNG folders.

Layout: <outdir>/{train,test}/<label>/<index>.png, plus <outdir>/classes.txt
(one class name per line, in label-index order).

Usage:
    python datagetters/cifar.py --variant 10 --outdir ./data/cifar10
    python datagetters/cifar.py --variant 100 --outdir ./data/cifar100
"""

import argparse
import pickle
import shutil
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

CIFAR_URL = "https://www.cs.toronto.edu/~kriz"
VARIANTS = {
    "10": {
        "archive": "cifar-10-python.tar.gz",
        "folder": "cifar-10-batches-py",
        "train": [f"data_batch_{i}" for i in range(1, 6)],
        "test": ["test_batch"],
        "meta": "batches.meta",
        "names_key": "label_names",
        "labels_key": "labels",
    },
    "100": {
        "archive": "cifar-100-python.tar.gz",
        "folder": "cifar-100-python",
        "train": ["train"],
        "test": ["test"],
        "meta": "meta",
        "names_key": "fine_label_names",
        "labels_key": "fine_labels",
    },
}


def _download(name: str, cache_dir: Path) -> Path:
    path = cache_dir / name
    if not path.exists():
        print(f"Downloading {name}")
        # Write to a temp name so an interrupted download isn't mistaken for a complete one.
        partial = path.with_suffix(path.suffix + ".part")
        urllib.request.urlretrieve(f"{CIFAR_URL}/{name}", partial)
        partial.rename(path)
    return path


def _unpickle(tar: tarfile.TarFile, member: str) -> dict:
    with tar.extractfile(member) as f:
        return pickle.load(f, encoding="latin1")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default="10")
    parser.add_argument("--outdir", type=Path, default=None)
    args = parser.parse_args()

    spec = VARIANTS[args.variant]
    outdir: Path = args.outdir or Path(f"data/cifar{args.variant}")
    cache_dir = outdir / "raw"
    cache_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(_download(spec["archive"], cache_dir), "r:gz") as tar:
        class_names = _unpickle(tar, f"{spec['folder']}/{spec['meta']}")[spec["names_key"]]
        (outdir / "classes.txt").write_text("\n".join(class_names) + "\n")

        for split in ("train", "test"):
            batches = [_unpickle(tar, f"{spec['folder']}/{name}") for name in spec[split]]
            # Rows are 3072 bytes: 1024 red, then green, then blue, each 32x32 row-major.
            images = np.concatenate([b["data"] for b in batches]).reshape(-1, 3, 32, 32)
            images = images.transpose(0, 2, 3, 1)
            labels = np.concatenate([b[spec["labels_key"]] for b in batches])

            split_dir = outdir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
            for label in range(len(class_names)):
                (split_dir / str(label)).mkdir(parents=True)

            for index, (image, label) in enumerate(
                tqdm(zip(images, labels), total=len(images), desc=split)
            ):
                # OpenCV expects BGR channel order.
                cv2.imwrite(
                    str(split_dir / str(label) / f"{index:05d}.png"),
                    cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                )


if __name__ == "__main__":
    main()
