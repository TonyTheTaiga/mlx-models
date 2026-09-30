"""
Export a trained CIFAR-10 ViT checkpoint for Lumen, the in-browser evidence viewer.

Writes to viz/vit/web/assets/:
  weights.bin / weights.json  raw float32 tensors + {name: [offset, shape]} manifest
  samples.bin / samples.json  uint8 test images (N, 32, 32, 3) + labels and MLX predictions
  reference.json              MLX logits for one sample per class, used by the page's self-check
  heatmaps.json / .bin        where the model looked, per gallery image (occlusion per patch,
                              gradient × input per pixel) and how faithful each method is

Usage: uv run python viz/vit/export.py [--checkpoint training/vit/checkpoints/cifar10]
"""

import argparse
import json
from pathlib import Path

import cv2
import mlx.core as mx
import mlx.nn as nn
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


def embed(model: Classifier, images: np.ndarray) -> mx.array:
    x = mx.array(images.astype(np.float32) / 255.0)
    cls = model.class_token(mx.zeros((len(images), 1), dtype=mx.int32))
    return mx.concatenate([cls, model.vit.proj(model.vit.preprocess(x))], axis=1)


def onehot(model: Classifier, target: np.ndarray) -> mx.array:
    return mx.array(np.eye(len(model.classifier.bias))[target], dtype=mx.float32)


def gradient_x_input(model: Classifier, images: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Per-pixel ∂logit/∂pixel · (pixel − image mean), summed over channels: (B, H, W).

    A first-order estimate of what flattening each pixel to the mean color (occlusion's
    baseline) would do to the target logit, from one backward pass.
    """
    x = mx.array(images.astype(np.float32) / 255.0)
    base = x.reshape(len(images), -1, 3).mean(1)[:, None, None, :]
    oh = onehot(model, target)
    grad = mx.grad(lambda xi: (model(xi) * oh).sum())(x)
    return np.array((grad * (x - base)).sum(-1))


def gradcam(model: Classifier, images: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Grad-CAM on the patch tokens entering the last block (the latest that reach the logit)."""
    blocks = model.vit.encoder_stack.layers
    x = embed(model, images)
    for block in blocks[:-1]:
        x = block(x)
    oh = onehot(model, target)
    grads = mx.grad(lambda t: (model.classifier(blocks[-1](t)[:, 0]) * oh).sum())(x)[:, 1:]
    return np.array(nn.relu((x[:, 1:] * grads.mean(1, keepdims=True)).sum(-1)))


def cls_attention(model: Classifier, images: np.ndarray) -> np.ndarray:
    """Last block's attention from CLS to each patch, averaged over heads."""
    blocks = model.vit.encoder_stack.layers
    x = embed(model, images)
    for block in blocks[:-1]:
        x = block(x)
    b = blocks[-1]
    q, k, _ = mx.split(b.proj(b.pn1(x)), 3, axis=-1)
    q, k = b.apply_rope(b.reshape(q)), b.apply_rope(b.reshape(k))
    att = mx.softmax((q @ k.transpose(0, 1, 3, 2)) * (b.d_model // b.n_heads) ** -0.5, axis=-1)
    return np.array(att[:, :, 0, 1:].mean(1))


def faithfulness(model, images, target, maps, grid, win):
    """
    Deletion / insertion AUC: mean p(target) as patches are flattened (deletion) or revealed on a
    flat image (insertion) hottest-first, over k = 0..P. Faithful maps: low deletion, high insertion.
    """
    P = grid * grid
    dele, ins = [], []
    for img, mp, t in zip(images, maps, target):
        mean = img.reshape(-1, 3).mean(0).round().astype(np.uint8)
        order = np.argsort(-np.asarray(mp))
        batch = []
        for k in range(P + 1):
            for reveal in (False, True):
                mask = np.zeros(P, bool)
                mask[order[:k]] = True
                if reveal:
                    mask = ~mask
                out = img.copy()
                out[np.kron(mask.reshape(grid, grid), np.ones((win, win), bool))] = mean
                batch.append(out)
        p = np.array(mx.softmax(model(mx.array(np.stack(batch).astype(np.float32) / 255.0)), -1))
        dele.append(p[0::2, t].mean())
        ins.append(p[1::2, t].mean())
    return {"deletion": float(np.mean(dele)), "insertion": float(np.mean(ins))}


def occlusion(model: Classifier, images: np.ndarray, target: np.ndarray, grid: int, win: int):
    """Drop in log p(target) when each patch is flattened to the image's mean color: (B, P)."""
    out = np.zeros((len(images), grid * grid), np.float32)
    for i, img in enumerate(images):
        mean = img.reshape(-1, 3).mean(0).round().astype(np.uint8)
        occ = np.repeat(img[None], grid * grid, axis=0)
        for p in range(grid * grid):
            r, c = divmod(p, grid)
            occ[p, r * win : (r + 1) * win, c * win : (c + 1) * win] = mean
        batch = mx.array(np.concatenate([img[None], occ]).astype(np.float32) / 255.0)
        lp = np.array(nn.log_softmax(model(batch), axis=-1))[:, target[i]]
        out[i] = lp[0] - lp[1:]
    return out


def load_model(ckpt: Path, weights: str = "best.npz") -> tuple[Classifier, dict, list[str]]:
    """The checkpoint's model (eval mode), its config, and the class names."""
    with open(ckpt / "config.yaml") as f:
        config = yaml.safe_load(f)
    dataset_config = config["dataset"]
    class_names = (ROOT / dataset_config["path"] / "classes.txt").read_text().split()
    model = Classifier(
        num_classes=len(class_names),
        image_size=dataset_config["image_size"],
        channels=dataset_config["channels"],
        **config["model"],
    )
    # Older checkpoints carry Rope2D tables that were (accidentally) trained; keep them so the
    # export reproduces the model as it was evaluated.
    load_checkpoint(model, ckpt / weights, learned_rope=True)
    model.eval()
    return model, config, class_names


def write_weights(model, config, class_names, ckpt_file: Path, out: Path) -> tuple[dict, int]:
    """weights.bin + weights.json for the browser model (model.js)."""
    # Rope2D's tables aren't parameters anymore, so add them explicitly: learned ones from old
    # checkpoints, the fixed defaults otherwise. The browser needs them either way.
    weights = {k: np.array(v, dtype=np.float32) for k, v in tree_flatten(model.parameters())}
    for i, block in enumerate(model.vit.encoder_stack.layers):
        prefix = f"vit.encoder_stack.layers.{i}.roper"
        weights[f"{prefix}.freq"] = np.array(block.roper.freq, dtype=np.float32)
        weights[f"{prefix}.pos_idx"] = np.array(block.roper.pos_idx, dtype=np.float32)
    manifest, offset = {}, 0
    with open(out / "weights.bin", "wb") as f:
        for name, value in sorted(weights.items()):
            manifest[name] = [offset, list(value.shape)]
            f.write(value.tobytes())
            offset += value.size
    with open(out / "weights.json", "w") as f:
        json.dump(
            {
                "config": config,
                "classes": class_names,
                "tensors": manifest,
                "fingerprint": fingerprint(ckpt_file),
            },
            f,
        )
    return manifest, offset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("training/vit/checkpoints/cifar10"))
    parser.add_argument("--weights", default="best.npz")
    parser.add_argument("--samples-per-class", type=int, default=16)
    args = parser.parse_args()

    ckpt = ROOT / args.checkpoint
    model, config, class_names = load_model(ckpt, args.weights)
    dataset_config = config["dataset"]
    data_root = ROOT / dataset_config["path"]
    OUT.mkdir(parents=True, exist_ok=True)
    manifest, offset = write_weights(model, config, class_names, ckpt / args.weights, OUT)

    # Gallery samples, with MLX's prediction for each so the gallery can mark mistakes.
    images, labels = load_split(data_root / "test", args.samples_per_class, seed=0)
    images.astype(np.uint8).tofile(OUT / "samples.bin")
    preds = np.array(mx.argmax(model(mx.array(images.astype(np.float32) / 255.0)), -1))
    with open(OUT / "samples.json", "w") as f:
        json.dump({"count": len(labels), "labels": labels.tolist(), "preds": preds.tolist()}, f)

    # Where the model looked for its prediction, per gallery image, and how faithful each
    # method is (Grad-CAM and CLS attention are scored for comparison but not shipped as maps).
    win = config["model"]["win_size"]
    grid = dataset_config["image_size"] // win
    occ = occlusion(model, images, preds, grid, win)
    gxi = gradient_x_input(model, images, preds)
    per_patch = lambda px: (
        px.reshape(len(px), grid, win, grid, win).sum((2, 4)).reshape(len(px), -1)
    )  # noqa: E731
    candidates = {
        "occlusion": occ,
        "gradient": per_patch(gxi),
        "gradcam": gradcam(model, images, preds),
        "cls_attention": cls_attention(model, images),
        "random": np.random.default_rng(0).random(occ.shape),
    }
    faith = {k: faithfulness(model, images, preds, v, grid, win) for k, v in candidates.items()}
    # Pixel-level gradient maps, positive part scaled per image to 0..255.
    pos = np.clip(gxi, 0, None)
    (pos / (pos.reshape(len(pos), -1).max(1)[:, None, None] + 1e-12) * 255).round().astype(
        np.uint8
    ).tofile(OUT / "heatmaps.bin")
    with open(OUT / "heatmaps.json", "w") as f:
        json.dump(
            {
                "fingerprint": fingerprint(ckpt / args.weights),
                "target": "predicted class",
                "occlusion": occ.round(5).tolist(),
                "gradient_patch": per_patch(gxi).round(6).tolist(),
                "faithfulness": faith,
            },
            f,
        )
    print("Faithfulness (deletion ↓ / insertion ↑):")
    for k, v in faith.items():
        print(f"  {k:<14} {v['deletion']:.3f} / {v['insertion']:.3f}")

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
