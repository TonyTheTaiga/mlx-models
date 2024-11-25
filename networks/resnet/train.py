import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import polars as pl
import numpy as np
import albumentations as A
import cv2


from model import ResNet, ResNet50

mx.random.seed(88)
np.random.seed(88)


transforms = A.Compose(
    [
        # A.RandomResizedCrop(height=384, width=384),
        A.RandomResizedCrop(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.46170458, 0.45680697, 0.42853157],
            std=[0.28604365, 0.2825701, 0.3046809],
        ),
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
    ]
)

val_transforms = A.Compose(
    [
        # A.SmallestMaxSize(438, interpolation=cv2.INTER_AREA, always_apply=True),
        # A.CenterCrop(384, 384, always_apply=True),
        A.SmallestMaxSize(256, interpolation=cv2.INTER_AREA, always_apply=True),
        A.CenterCrop(224, 224, always_apply=True),
        A.Normalize(
            mean=[0.46170458, 0.45680697, 0.42853157],
            std=[0.28604365, 0.2825701, 0.3046809],
        ),
    ]
)


def ce_loss_fn(model, inputs, labels):
    logits = model(inputs)
    return nn.losses.cross_entropy(
        logits, labels, reduction="mean", label_smoothing=0.1
    )


def focal_loss(model, inputs, labels): ...


def load_image(image_path, augment):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    if augment:
        image = transforms(image=image)["image"]
    else:
        image = val_transforms(image=image)["image"]

    return mx.array(image)


def batch_iterate(batch_size, X, y, augment):
    perm = mx.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield (
            mx.stack([load_image(path, augment) for path in X[ids]], axis=0),
            mx.array(y[ids]),
        )


def val_loop(model, frame):
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    for inputs, targets in batch_iterate(
        4, frame["image"].to_numpy(), frame["target"].to_numpy(), False
    ):
        batch_size = len(inputs)
        logits = model(inputs)

        batch_loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
        total_loss += batch_loss.item() * batch_size  # pyright: ignore

        accuracy = mx.mean(mx.argmax(logits, axis=1) == targets)  # pyright: ignore
        total_correct += accuracy.item() * batch_size  # pyright: ignore
        total_samples += batch_size

    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    avg_accuracy = total_correct / total_samples if total_samples > 0 else 0.0

    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
    return avg_accuracy


def train_loop(model, df, optimizer, loss_and_grad_fn):
    train_loss = 0
    iter_count = 0
    total_samples = 0
    for inputs, targets in batch_iterate(
        4, df["image"].to_numpy(), df["target"].to_numpy(), True
    ):
        batch_size = len(inputs)
        loss, grads = loss_and_grad_fn(model, inputs, targets)
        train_loss += loss.item() * batch_size
        total_samples += batch_size
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        iter_count += 1
        if iter_count % 250 == 0:
            print(
                f"{iter_count} loss:",
                train_loss / total_samples,
            )

    return train_loss / total_samples


def get_cosine_lr(base_lr, current_epoch, total_epochs, warmup_epochs):
    if current_epoch < warmup_epochs:
        return base_lr * (current_epoch + 1) / warmup_epochs

    progress = (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    model = ResNet50(3, num_classes=10)
    mx.eval(model.parameters())
    loss_and_grad_fn = nn.value_and_grad(model, ce_loss_fn)
    df = pl.read_csv("train.csv", has_header=False, new_columns=["image", "target"])
    val_df = pl.read_csv("val.csv", has_header=False, new_columns=["image", "target"])
    base_lr = 0.01
    optimizer = optim.SGD(base_lr, momentum=0.875, weight_decay=1 / 32768)
    epochs = 200
    curr_best = 0.0
    for epoch in range(epochs):
        optimizer.learning_rate = get_cosine_lr(base_lr, epoch, epochs, 3)
        train_loss = train_loop(model, df, optimizer, loss_and_grad_fn)
        print(f"epoch {epoch} -- loss: {train_loss}")
        acc = val_loop(model, val_df)
        if acc > curr_best:
            curr_best = acc
            model.save_weights(f"224/imagenette_conv50@{epoch}.npz")

    model.save_weights("224/imagenette_conv50.npz")


if __name__ == "__main__":
    main()

