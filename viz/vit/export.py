"""
Export a trained CIFAR-10 ViT checkpoint for the browser visualizer.

Writes to viz/vit/web/assets/:
  weights.bin / weights.json  raw float32 tensors + {name: [offset, shape]} manifest
  samples.bin / samples.json  uint8 test images (N, 32, 32, 3) + labels and class names
  pca.json                    per-depth PCA basis of the residual stream (for false color)
  pca_mid.json                the same for each block's state between attention and the MLP
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


def after_attention(block, x: mx.array) -> mx.array:
    """The first half of EncoderBlock.__call__: x plus its attention update."""
    q, k, v = mx.split(block.proj(block.pn1(x)), 3, axis=-1)
    q = block.apply_rope(block.reshape(q))
    k = block.apply_rope(block.reshape(k))
    scale = (block.d_model // block.n_heads) ** -0.5
    attention = mx.fast.scaled_dot_product_attention(q, k, block.reshape(v), scale=scale)
    return x + block.atten_proj(attention.transpose(0, 2, 1, 3).reshape(*x.shape))


def residual_states(
    model: Classifier, images: mx.array
) -> tuple[list[mx.array], list[mx.array], mx.array]:
    """
    Token states after the patch embedding and after every encoder block, plus each block's
    state between attention and the MLP.
    """
    patches = model.vit.preprocess(images)
    cls = model.class_token(mx.zeros((images.shape[0], 1), dtype=mx.int32))
    x = mx.concatenate([cls, model.vit.proj(patches)], axis=1)
    states, mids = [x], []
    for block in model.vit.encoder_stack.layers:
        mid = after_attention(block, x)
        x = mid + block.feed_forward(block.pn2(mid))
        mids.append(mid)
        states.append(x)
    return states, mids, model.classifier(x[:, 0])


def fit_pca(tokens: np.ndarray, align: np.ndarray | None = None) -> dict:
    """
    Top-3 PCA basis with 2–98th percentile ranges. With `align`, components are sign-flipped
    to agree with that basis, so neighbouring states get comparable colors.
    """
    mean = tokens.mean(0)
    _, _, vt = np.linalg.svd(tokens - mean, full_matrices=False)
    comps = vt[:3]
    if align is not None:
        comps = comps * np.sign(np.sum(comps * align, axis=1, keepdims=True) + 1e-12)
    proj = (tokens - mean) @ comps.T
    lo, hi = np.percentile(proj, 2, axis=0), np.percentile(proj, 98, axis=0)
    return {
        "mean": mean.tolist(),
        "components": comps.tolist(),
        "lo": lo.tolist(),
        "hi": hi.tolist(),
    }


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
        json.dump({"config": config, "classes": class_names, "tensors": manifest}, f)

    # Gallery samples.
    images, labels = load_split(data_root / "test", args.samples_per_class, seed=0)
    images.astype(np.uint8).tofile(OUT / "samples.bin")
    with open(OUT / "samples.json", "w") as f:
        json.dump({"count": len(labels), "labels": labels.tolist()}, f)

    # Per-depth PCA over a larger pool, so colors are stable across images.
    pool, _ = load_split(data_root / "test", 100, seed=1)
    n_layers = config["model"]["n_layers"]
    per_depth: list[list[np.ndarray]] = [[] for _ in range(n_layers + 1)]
    per_mid: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    for start in range(0, len(pool), 250):
        batch = mx.array(pool[start : start + 250].astype(np.float32) / 255.0)
        states, mids, _ = residual_states(model, batch)
        for depth, s in enumerate(states):
            per_depth[depth].append(np.array(s[:, 1:]).reshape(-1, s.shape[-1]))
        for depth, s in enumerate(mids):
            per_mid[depth].append(np.array(s[:, 1:]).reshape(-1, s.shape[-1]))
    pca = [fit_pca(np.concatenate(tokens)) for tokens in per_depth]
    # Mid-block colors are aligned to the block's output basis, so the MLP step reads as a
    # change in color rather than an arbitrary palette swap.
    pca_mid = [
        fit_pca(np.concatenate(tokens), align=np.array(pca[i + 1]["components"]))
        for i, tokens in enumerate(per_mid)
    ]
    with open(OUT / "pca.json", "w") as f:
        json.dump(pca, f)
    with open(OUT / "pca_mid.json", "w") as f:
        json.dump(pca_mid, f)

    # Reference outputs for the browser self-check: the first gallery image of every class.
    indices = [int(np.flatnonzero(labels == c)[0]) for c in np.unique(labels)]
    ref = mx.array(images[indices].astype(np.float32) / 255.0)
    states, mids, logits = residual_states(model, ref)
    assert mx.allclose(logits, model(ref), atol=1e-4).item()  # the split forward matches
    with open(OUT / "reference.json", "w") as f:
        json.dump({"indices": indices, "logits": np.array(logits).tolist()}, f)

    acc = (np.array(mx.argmax(model(mx.array(images.astype(np.float32) / 255.0)), -1)) == labels)
    print(f"Exported {len(manifest)} tensors ({offset * 4 / 1e6:.1f} MB), {len(labels)} samples")
    print(f"Gallery accuracy: {acc.mean():.1%}")


if __name__ == "__main__":
    main()
