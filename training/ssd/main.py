from __future__ import annotations
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from networks.ssd.model import SSD300
from networks.ssd.utils import generate_anchors, load_data, prepare_ssd_dataset

DATASET_ROOT = Path("/Users/taigaishida/workspace/mlx-models/pedestrians/")


def dataloader(data, batch_size):
    idx = mx.random.permutation(len(data[0]))
    for start in range(0, len(data[0]), batch_size):
        yield (
            data[0][idx[start : start + batch_size]],
            data[1][idx[start : start + batch_size]],
            data[2][idx[start : start + batch_size]],
        )


def loss_fn(model, images: mx.array, loc_targets: mx.array, cls_targets: mx.array, alpha: float = 1.0):
    loc_preds, cls_preds = model(images)
    loc_loss = mx.mean(nn.losses.huber_loss(loc_preds, loc_targets))
    cls_loss = mx.mean(nn.losses.cross_entropy(cls_preds, cls_targets))
    return cls_loss + alpha * loc_loss


def main():
    learning_rate = 1e-3

    data = load_data(DATASET_ROOT, 300)
    anchors = generate_anchors([1, 2, 3, 1 / 2, 1 / 3], feature_map_sizes=[37, 18, 9, 5, 3, 1])
    dataset = prepare_ssd_dataset(data, anchors)
    model = SSD300(num_classes=2)  # pedestrian + background

    # load vgg16 weights
    model.load_weights(
        "/Users/taigaishida/workspace/mlx-models/networks/vgg16/weights.npz",
        strict=False,
    )
    mx.eval(model.parameters())
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loc_optim = optim.AdamW(learning_rate=learning_rate)
    cls_optim = optim.AdamW(learning_rate=learning_rate)

    image = mx.expand_dims(data[0]["resized_image"], 0)
    pred_loc, pred_cls = model(image)

    for epoch in range(5):
        culm_loss = 0
        num_samples = 0
        for images, loc_targets, cls_targets in dataloader(dataset, batch_size=4):
            loss, grads = loss_and_grad_fn(model, images, loc_targets, cls_targets)
            loc_optim.update(model, grads)
            cls_optim.update(model, grads)
            mx.eval(model.parameters(), loc_optim.state, cls_optim.state)
            culm_loss += loss.item() * (images.shape[0])
            num_samples += images.shape[0]

        print(f"train loss @{epoch}", culm_loss / num_samples)


if __name__ == "__main__":
    main()
