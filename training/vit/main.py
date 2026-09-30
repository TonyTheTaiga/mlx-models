import argparse
import math
from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml

from networks.transformers.vit.model import Vit


class ImageFolder:
    """Images laid out as <root>/<split>/<label>/*.png, with integer label folder names."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        batch_size: int,
        channels: int,
        shuffle: bool = True,
    ) -> None:
        split_path = Path(root) / split
        if split not in {"train", "test"} or not split_path.is_dir():
            raise ValueError(f"Invalid split: {split_path}")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if channels not in {1, 3}:
            raise ValueError("channels must be 1 or 3")

        self.batch_size = batch_size
        self.channels = channels
        self.shuffle = shuffle
        labels = sorted(int(d.name) for d in split_path.iterdir() if d.name.isdigit())
        if labels != list(range(len(labels))):
            raise ValueError(f"Label folders under {split_path} must be 0..N-1, got {labels}")
        self.num_classes = len(labels)
        self.samples = [
            (path, label)
            for label in labels
            for path in sorted((split_path / str(label)).glob("*.png"))
        ]
        if not self.samples:
            raise ValueError(f"No images found under {split_path}")

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
                if self.channels == 1:
                    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                else:
                    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Failed to decode {path}")
                if self.channels == 1:
                    image = image[:, :, None]
                else:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)

            yield (
                mx.array(np.stack(images).astype(np.float32) / 255.0),
                mx.array([label for _, label in batch], dtype=mx.int32),
            )


class Classifier(nn.Module):
    def __init__(self, num_classes: int, **vit_config) -> None:
        super().__init__()
        self.vit = Vit(**vit_config)
        self.class_token = nn.Embedding(num_embeddings=1, dims=vit_config["d_model"])
        self.classifier = nn.Linear(vit_config["d_model"], num_classes)

    def __call__(self, images: mx.array) -> mx.array:
        patches = self.vit.preprocess(images)
        class_indices = mx.zeros((images.shape[0], 1), dtype=mx.int32)
        encoded = self.vit(patches, prefix_tokens=self.class_token(class_indices))
        return self.classifier(encoded[:, 0])


def load_checkpoint(model: nn.Module, path: str | Path, learned_rope: bool = False) -> dict:
    """
    Load weights, tolerating checkpoints saved while Rope2D's freq/pos_idx were still
    (accidentally) trainable. Those `*.roper.{freq,pos_idx}` entries are dropped unless
    `learned_rope`, in which case they overwrite the fixed defaults. Returns the roper entries.
    """
    weights = mx.load(str(path))
    rope = {k: v for k, v in weights.items() if k.split(".")[-2:-1] == ["roper"]}
    model.load_weights([(k, v) for k, v in weights.items() if k not in rope])
    if learned_rope:
        modules = dict(model.named_modules())
        for key, value in rope.items():
            owner, name = key.rsplit(".", 1)
            setattr(modules[owner], f"_{name}", value)
    return rope


def loss_fn(model: Classifier, images: mx.array, labels: mx.array) -> mx.array:
    logits = model(images)
    return nn.losses.cross_entropy(logits, labels, reduction="mean")


def evaluate(model: Classifier, dataset: ImageFolder) -> tuple[float, float]:
    total_loss = 0.0
    correct = 0
    for images, labels in dataset:
        logits = model(images)
        loss = nn.losses.cross_entropy(logits, labels, reduction="sum")
        hits = (mx.argmax(logits, axis=-1) == labels).sum()
        mx.eval(loss, hits)
        total_loss += loss.item()
        correct += hits.item()
    return total_loss / len(dataset.samples), correct / len(dataset.samples)


def make_schedule(training_config: dict, total_steps: int) -> float | Callable:
    peak = training_config["learning_rate"]
    warmup_steps = training_config.get("warmup_steps", 0)
    if training_config.get("cosine_decay", False):
        schedule = optim.cosine_decay(peak, total_steps - warmup_steps)
    else:
        schedule = peak
    if not warmup_steps:
        return schedule
    warmup = optim.linear_schedule(0.0, peak, warmup_steps)
    if not callable(schedule):
        return warmup  # linear_schedule holds at `peak` once warmup_steps is reached
    return optim.join_schedules([warmup, schedule], [warmup_steps])


def main():
    parser = argparse.ArgumentParser(description="Train a ViT image classifier.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "config.yaml",
        help="Config file, e.g. training/vit/config_cifar10.yaml",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    dataset_config = config["dataset"]
    root = Path(__file__).parents[2] / dataset_config["path"]
    train = ImageFolder(
        root=root,
        split="train",
        batch_size=config["training"]["batch_size"],
        channels=dataset_config["channels"],
    )
    test = ImageFolder(
        root=root,
        split="test",
        batch_size=config["training"]["batch_size"],
        channels=dataset_config["channels"],
        shuffle=False,
    )

    model = Classifier(
        num_classes=train.num_classes,
        image_size=dataset_config["image_size"],
        channels=dataset_config["channels"],
        **config["model"],
    )
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    epochs = config["training"]["epochs"]
    steps = len(train)

    optimizer = optim.Adam(learning_rate=make_schedule(config["training"], epochs * steps))

    checkpoint_dir = None
    if config.get("checkpoint_dir"):
        checkpoint_dir = Path(__file__).parents[2] / config["checkpoint_dir"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_dir / "config.yaml", "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    best_accuracy = -1.0

    def one_epoch(epoch_idx: int) -> list[float]:
        losses = []
        for step, (images, labels) in enumerate(train):
            loss, grads = loss_and_grad_fn(model, images, labels)
            optimizer.update(model, grads)
            mx.eval(loss, model.parameters(), optimizer.state)
            loss_value = loss.item()
            losses.append(loss_value)
            print(
                f"Epoch {epoch_idx + 1}/{epochs} step {step + 1}/{steps} — loss: {loss_value:.4f}"
            )
        return losses

    for epoch_idx in range(epochs):
        losses = one_epoch(epoch_idx)
        print(
            f"Epoch {epoch_idx + 1}/{epochs} — mean loss: {sum(losses) / len(losses):.4f} "
            f"({losses[0]:.4f} -> {losses[-1]:.4f})"
        )
        test_loss, test_accuracy = evaluate(model, test)
        print(
            f"Epoch {epoch_idx + 1}/{epochs} — test loss: {test_loss:.4f}, "
            f"test accuracy: {test_accuracy:.2%}"
        )

        if checkpoint_dir is not None:
            model.save_weights(str(checkpoint_dir / "last.npz"))
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                model.save_weights(str(checkpoint_dir / "best.npz"))
                print(f"Saved new best checkpoint ({test_accuracy:.2%}) to {checkpoint_dir}")


if __name__ == "__main__":
    main()
