import math
from pathlib import Path

import cv2
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten
from tqdm import tqdm

from networks.mobilenet.model import PAPER_CONFIG, MobileNet

MNIST_PATH = Path("/Users/taigaishida/workspace/mlx-models/mnist/")
IMAGE_SIZE = 224
NUM_CLASSES = 10
GRAD_CLIP_NORM = 1.0
GLOBAL_SEED = 42

mx.random.seed(GLOBAL_SEED)


def _clip_gradients(grads, clip_norm: float):
    if not clip_norm or clip_norm <= 0:
        return grads

    flat = dict(tree_flatten(grads))
    total = mx.array(0.0, dtype=mx.float32)
    for g in flat.values():
        g32 = g.astype(mx.float32)
        total = total + mx.sum(g32 * g32)
    norm = mx.sqrt(total + 1e-12)
    scale = mx.minimum(
        mx.array(1.0, dtype=mx.float32), mx.array(clip_norm, dtype=mx.float32) / (norm + 1e-12)
    )
    clipped = {k: (v.astype(mx.float32) * scale).astype(v.dtype) for k, v in flat.items()}
    return tree_unflatten(clipped)


def read_image(path: Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Load, resize, and normalize an RGB image."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {path}")

    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32) / 255.0


def load_split(split_root: Path) -> tuple[list[Path], list[int]]:
    file_paths: list[Path] = []
    labels: list[int] = []

    for class_dir in sorted(split_root.iterdir()):
        if not class_dir.is_dir() or not class_dir.name.isdigit():
            continue

        class_id = int(class_dir.name)
        for path in sorted(class_dir.rglob("*.png")):
            file_paths.append(path)
            labels.append(class_id)

    if not file_paths:
        raise RuntimeError(f"No images found under {split_root}")

    return file_paths, labels


def load_mnist() -> dict[str, list]:
    train_files, train_labels = load_split(MNIST_PATH / "training")
    val_files, val_labels = load_split(MNIST_PATH / "testing")

    return {
        "train_files": train_files,
        "train_labels": train_labels,
        "val_files": val_files,
        "val_labels": val_labels,
    }


def dataloader(
    file_paths: list[Path],
    labels: list[int],
    batch_size: int,
    shuffle: bool = True,
    seed: int | None = None,
):
    if not file_paths:
        return

    indices = np.arange(len(file_paths))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch_images = [read_image(file_paths[i]) for i in batch_idx]
        batch_labels = [labels[i] for i in batch_idx]
        yield (
            mx.array(np.stack(batch_images, axis=0), dtype=mx.float32),
            mx.array(np.array(batch_labels, dtype=np.int32)),
        )


def loss_fn(model: MobileNet, images: mx.array, labels: mx.array):
    logits = model(images)
    loss = nn.losses.cross_entropy(logits, labels, reduction="mean")
    return loss, logits


def train_epoch(
    model: MobileNet,
    optimizer,
    file_paths: list[Path],
    labels: list[int],
    batch_size: int,
    loss_and_grad_fn,
    epoch: int,
    total_epochs: int,
    seed: int,
):
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    num_steps = math.ceil(len(file_paths) / batch_size)
    progress = tqdm(
        dataloader(file_paths, labels, batch_size, shuffle=True, seed=seed),
        total=num_steps,
        desc=f"Epoch {epoch}/{total_epochs} [train]",
        leave=False,
    )
    for images, batch_labels in progress:
        (loss, logits), grads = loss_and_grad_fn(model, images, batch_labels)
        grads = _clip_gradients(grads, GRAD_CLIP_NORM)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)

        batch_count = images.shape[0]
        total_loss += loss.item() * batch_count
        total_correct += mx.sum(mx.equal(mx.argmax(logits, axis=-1), batch_labels)).item()
        total_samples += batch_count

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        progress.set_postfix(
            loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}", avg_acc=f"{avg_acc:.4f}"
        )

    return total_loss / total_samples, total_correct / total_samples


def eval_epoch(
    model: MobileNet,
    file_paths: list[Path],
    labels: list[int],
    batch_size: int,
    epoch: int,
    total_epochs: int,
):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    num_steps = math.ceil(len(file_paths) / batch_size)
    progress = tqdm(
        dataloader(file_paths, labels, batch_size, shuffle=False),
        total=num_steps,
        desc=f"Epoch {epoch}/{total_epochs} [val]",
        leave=False,
    )
    for images, batch_labels in progress:
        logits = model(images)
        loss = nn.losses.cross_entropy(logits, batch_labels, reduction="mean")

        batch_count = images.shape[0]
        total_loss += loss.item() * batch_count
        total_correct += mx.sum(mx.equal(mx.argmax(logits, axis=-1), batch_labels)).item()
        total_samples += batch_count

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        progress.set_postfix(
            loss=f"{loss.item():.4f}", avg_loss=f"{avg_loss:.4f}", avg_acc=f"{avg_acc:.4f}"
        )

    return total_loss / total_samples, total_correct / total_samples


def main():
    batch_size = 32
    val_batch_size = 32
    epochs = 15
    learning_rate = 2e-4

    dataset = load_mnist()
    model = MobileNet(num_classes=NUM_CLASSES, config=PAPER_CONFIG)
    mx.eval(model.parameters())

    optimizer = optim.Adam(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            optimizer,
            dataset["train_files"],
            dataset["train_labels"],
            batch_size,
            loss_and_grad_fn,
            epoch,
            epochs,
            seed=GLOBAL_SEED + epoch,
        )
        val_loss, val_acc = eval_epoch(
            model,
            dataset["val_files"],
            dataset["val_labels"],
            val_batch_size,
            epoch,
            epochs,
        )

        print(
            f"Epoch {epoch}/{epochs} "
            f"- train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )


if __name__ == "__main__":
    main()
