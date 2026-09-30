"""
Export a trained CIFAR-10 ViT checkpoint for Lumen, the in-browser evidence viewer.

Writes to viz/vit/web/assets/:
  weights.bin / weights.json  raw float32 tensors + {name: [offset, shape]} manifest
  samples.bin / samples.json  uint8 test images (N, 32, 32, 3) + labels and MLX predictions
  reference.json              MLX logits for one sample per class, used by the page's self-check

Usage: uv run python viz/vit/export.py [--checkpoint training/vit/checkpoints/cifar10]
"""

import argparse
import json
from pathlib import Path

import cv2
import mlx.core as mx
import numpy as np
import yaml
from mlx.utils import tree_flatten

from training.vit.main import Classifier, load_checkpoint

ROOT = Path(__file__).parents[2]
OUT = Path(__file__).parent / "web" / "assets"


def fingerprint(path: Path) -> str:
    """Short content hash of a checkpoint, so the page can tell if its files disagree."""
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def load_split(split_dir: Path, per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for label in sorted(int(d.name) for d in split_dir.iterdir() if d.name.isdigit()):
        paths = sorted((split_dir / str(label)).glob("*.png"))
        for path in rng.choice(paths, size=min(per_class, len(paths)), replace=False):
            image = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            images.append(image)
            labels.append(label)
    return np.stack(images), np.array(labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("training/vit/checkpoints/cifar10"))
    parser.add_argument("--weights", default="best.npz")
    parser.add_argument("--samples-per-class", type=int, default=16)
    args = parser.parse_args()

    ckpt = ROOT / args.checkpoint
    with open(ckpt / "config.yaml") as f:
        config = yaml.safe_load(f)
    dataset_config = config["dataset"]
    data_root = ROOT / dataset_config["path"]
    class_names = (data_root / "classes.txt").read_text().split()

    model = Classifier(
        num_classes=len(class_names),
        image_size=dataset_config["image_size"],
        channels=dataset_config["channels"],
        **config["model"],
    )
    # Older checkpoints carry Rope2D tables that were (accidentally) trained; keep them so the
    # export reproduces the model as it was evaluated.
    load_checkpoint(model, ckpt / args.weights, learned_rope=True)
    model.eval()
    OUT.mkdir(parents=True, exist_ok=True)

    # Rope2D's tables aren't parameters anymore, so add them explicitly: learned ones from old
    # checkpoints, the fixed defaults otherwise. The browser needs them either way.
    weights = {k: np.array(v, dtype=np.float32) for k, v in tree_flatten(model.parameters())}
    for i, block in enumerate(model.vit.encoder_stack.layers):
        prefix = f"vit.encoder_stack.layers.{i}.roper"
        weights[f"{prefix}.freq"] = np.array(block.roper.freq, dtype=np.float32)
        weights[f"{prefix}.pos_idx"] = np.array(block.roper.pos_idx, dtype=np.float32)
    manifest, offset = {}, 0
    with open(OUT / "weights.bin", "wb") as f:
        for name, value in sorted(weights.items()):
            manifest[name] = [offset, list(value.shape)]
            f.write(value.tobytes())
            offset += value.size
    with open(OUT / "weights.json", "w") as f:
        json.dump({"config": config, "classes": class_names, "tensors": manifest, "fingerprint": fingerprint(ckpt / args.weights)}, f)

    # Gallery samples, with MLX's prediction for each so the gallery can mark mistakes.
    images, labels = load_split(data_root / "test", args.samples_per_class, seed=0)
    images.astype(np.uint8).tofile(OUT / "samples.bin")
    preds = np.array(mx.argmax(model(mx.array(images.astype(np.float32) / 255.0)), -1))
    with open(OUT / "samples.json", "w") as f:
        json.dump({"count": len(labels), "labels": labels.tolist(), "preds": preds.tolist()}, f)

    # Reference outputs for the browser self-check: the first gallery image of every class.
    indices = [int(np.flatnonzero(labels == c)[0]) for c in np.unique(labels)]
    ref = mx.array(images[indices].astype(np.float32) / 255.0)
    logits = model(ref)
    with open(OUT / "reference.json", "w") as f:
        json.dump({"indices": indices, "logits": np.array(logits).tolist()}, f)

    acc = preds == labels
    print(f"Exported {len(manifest)} tensors ({offset * 4 / 1e6:.1f} MB), {len(labels)} samples")
    print(f"Gallery accuracy: {acc.mean():.1%}")


if __name__ == "__main__":
    main()
