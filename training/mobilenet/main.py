from pathlib import Path
from uuid import uuid4

import cv2
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from tora import Tora

from networks.mobilenet.model import PAPER_CONFIG, MobileNet

MNIST_PATH = Path("/Users/taigaishida/workspace/mlx-models/mnist/")
IMAGE_SIZE = 224
NUM_CLASSES = 10
MAX_BATCHES_IN_MEMORY = 4


def read_image(path: Path, image_size: int = IMAGE_SIZE) -> np.ndarray:
    """Load, resize, normalize, and tile MNIST grayscale image into 3 channels."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to read image at {path}")

    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = np.stack([img, img, img], axis=-1)  # repeat grayscale channel
    return img


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


def dataloader(file_paths: list[Path], labels: list[int], batch_size: int, shuffle: bool = True):
    num_samples = len(file_paths)
    order = np.arange(num_samples)
    if shuffle:
        rng = np.random.default_rng()
        rng.shuffle(order)

    chunk_size = batch_size * MAX_BATCHES_IN_MEMORY

    for chunk_start in range(0, num_samples, chunk_size):
        chunk_idx = order[chunk_start : chunk_start + chunk_size]
        chunk_images = [read_image(file_paths[i]) for i in chunk_idx]
        chunk_labels = [labels[i] for i in chunk_idx]

        if not chunk_images:
            continue

        chunk_images_arr = mx.array(np.stack(chunk_images, axis=0), dtype=mx.float32)
        chunk_labels_arr = mx.array(np.array(chunk_labels, dtype=np.int32))

        for start in range(0, chunk_images_arr.shape[0], batch_size):
            end = min(start + batch_size, chunk_images_arr.shape[0])
            yield chunk_images_arr[start:end], chunk_labels_arr[start:end]


def loss_fn(model: MobileNet, images: mx.array, labels: mx.array):
    logits = model(images)
    loss = nn.losses.cross_entropy(logits, labels, reduction="mean")
    return loss, logits


def evaluate(model: MobileNet, file_paths: list[Path], labels: list[int], batch_size: int):
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    for batch_images, batch_labels in dataloader(file_paths, labels, batch_size, shuffle=False):
        logits = model(batch_images)
        loss = nn.losses.cross_entropy(logits, batch_labels, reduction="mean").item()
        preds = mx.argmax(logits, axis=-1)
        total_loss += loss * batch_images.shape[0]
        total_correct += mx.sum(mx.equal(preds, batch_labels)).item()
        total_samples += batch_images.shape[0]

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    return avg_loss, accuracy


def main():
    batch_size = 64
    val_batch_size = 128
    epochs = 15
    learning_rate = 1e-3

    dataset = load_mnist()
    model = MobileNet(num_classes=NUM_CLASSES, config=PAPER_CONFIG)
    mx.eval(model.parameters())

    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))

    optimizer = optim.AdamW(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    tora = Tora.create_experiment(
        name=f"MobileNet_MNIST_{uuid4().hex[:3]}",
        description="MobileNet v2 on MNIST",
        hyperparams={
            "architecture": "MobileNet",
            "image_size": IMAGE_SIZE,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "optimizer": "AdamW",
            "num_trainable_params": num_params,
        },
        max_buffer_len=1,
        workspace_id="f6ed548d-21e9-4f0b-a065-72103c775a3d",
    )

    for epoch in range(epochs):
        culm_loss = 0.0
        correct = 0.0
        num_samples = 0

        for images, labels in dataloader(
            dataset["train_files"], dataset["train_labels"], batch_size
        ):
            (loss, logits), grads = loss_and_grad_fn(model, images, labels)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            culm_loss += loss.item() * images.shape[0]
            correct += mx.sum(mx.equal(mx.argmax(logits, axis=-1), labels)).item()
            num_samples += images.shape[0]

        train_loss = culm_loss / num_samples
        train_acc = correct / num_samples

        val_loss, val_acc = evaluate(
            model, dataset["val_files"], dataset["val_labels"], val_batch_size
        )

        tora.metric(name="train_loss", step_or_epoch=epoch, value=train_loss)
        tora.metric(name="train_accuracy", step_or_epoch=epoch, value=train_acc)
        tora.metric(name="val_loss", step_or_epoch=epoch, value=val_loss)
        tora.metric(name="val_accuracy", step_or_epoch=epoch, value=val_acc)
        tora.metric(name="lr", step_or_epoch=epoch, value=learning_rate)

        print(
            f"Epoch {epoch + 1}/{epochs} "
            f"- train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

    tora.shutdown()


if __name__ == "__main__":
    main()
