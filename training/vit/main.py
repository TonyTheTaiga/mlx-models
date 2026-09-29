import math
from collections.abc import Iterator
from pathlib import Path

import cv2
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml

from networks.transformers.vit.model import Vit


class MNIST:
    def __init__(self, root: str | Path, split: str, batch_size: int, shuffle: bool = True) -> None:
        split_path = Path(root) / split
        if split not in {"train", "test"} or not split_path.is_dir():
            raise ValueError(f"Invalid MNIST split: {split_path}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.samples = [
            (path, label)
            for label in range(10)
            for path in sorted((split_path / str(label)).glob("*.png"))
        ]
        if not self.samples:
            raise ValueError(f"No MNIST images found under {split_path}")

    def __len__(self) -> int:
        return math.ceil(len(self.samples) / self.batch_size)

    def __iter__(self) -> Iterator[tuple[mx.array, mx.array]]:
        indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch = [self.samples[index] for index in indices[start : start + self.batch_size]]
            images = []
            for path, _ in batch:
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"Failed to decode {path}")
                images.append(image)

            yield (
                mx.array(np.stack(images).astype(np.float32)[:, :, :, None] / 255.0),
                mx.array([label for _, label in batch], dtype=mx.int32),
            )


class MNISTClassifier(nn.Module):
    def __init__(self, **model_config) -> None:
        super().__init__()
        self.vit = Vit(**model_config)
        self.class_token = nn.Embedding(num_embeddings=1, dims=model_config["d_model"])
        self.classifier = nn.Linear(model_config["d_model"], 10)

    def __call__(self, images: mx.array) -> mx.array:
        patches = self.vit.preprocess(images)
        class_indices = mx.zeros((images.shape[0], 1), dtype=mx.int32)
        encoded = self.vit(patches, prefix_tokens=self.class_token(class_indices))
        return self.classifier(encoded[:, 0])

def loss_fn(model: MNISTClassifier, images: mx.array, labels: mx.array) -> mx.array:
    logits = model(images)
    return nn.losses.cross_entropy(logits, labels, reduction="mean")


def main():
    with open(Path(__file__).parent / "config.yaml") as f:
        config = yaml.safe_load(f)

    model = MNISTClassifier(**config["model"])
    optimizer = optim.Adam(learning_rate=config["training"]["learning_rate"])
    train = MNIST(
        root=Path(__file__).parents[2] / config["dataset"]["path"],
        split="train",
        batch_size=config["training"]["batch_size"],
    )
    images, labels = next(iter(train))

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    steps = int(config["training"].get("steps", 3))
    losses = []
    for step in range(steps):
        loss, grads = loss_and_grad_fn(model, images, labels)
        loss_value = loss.item()
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(loss_value)
        print(f"Step {step + 1}/{steps} — loss: {loss_value:.4f}")

    if len(losses) > 1:
        print(f"Loss change: {losses[0]:.4f} -> {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
