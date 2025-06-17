from __future__ import annotations
from pathlib import Path
from uuid import uuid4
import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from networks.ssd.model import SSD300
from networks.ssd.utils import generate_anchors, load_data, prepare_ssd_dataset
from utils import decode_predictions, visualize_detections
import numpy as np

from tora import Tora # pyright: ignore

DATASET_ROOT = Path("/Users/taigaishida/workspace/mlx-models/pedestrians/")


def dataloader(data, batch_size):
    idx = mx.random.permutation(len(data[0]))
    for start in range(0, len(data[0]), batch_size):
        yield (
            data[0][idx[start : start + batch_size]],
            data[1][idx[start : start + batch_size]],
            data[2][idx[start : start + batch_size]],
        )


def loss_fn(model, images, loc_targets, cls_targets, alpha=1.0):
    loc_preds, cls_preds = model(images)
    B, N, C = cls_preds.shape

    pos = cls_targets > 0
    neg = ~pos

    # --- localisation -------------------------------------------------------
    loc_loss_all = nn.losses.huber_loss(loc_preds, loc_targets, reduction="none")
    loc_loss = mx.sum(mx.where(mx.expand_dims(pos, -1), loc_loss_all, 0.0))
    loc_loss = alpha * loc_loss / mx.maximum(mx.sum(pos).astype(mx.float32), 1.0)

    # --- classification + hard-neg mining -----------------------------------
    cls_loss_all = nn.losses.cross_entropy(cls_preds.reshape(-1, C), cls_targets.reshape(-1), reduction="none").reshape(
        B, N
    )

    scores = mx.stop_gradient(cls_loss_all * neg)  # <— detach
    ranked = mx.argsort(scores, axis=1)
    k = mx.minimum(3 * mx.sum(pos, axis=1), mx.sum(neg, axis=1))
    arange_N = mx.arange(N)

    hard_neg = arange_N[None, :] >= (N - k[:, None])
    hard_neg = mx.take_along_axis(hard_neg, ranked, axis=1)

    sel = pos | hard_neg
    cls_loss = mx.sum(cls_loss_all * sel) / mx.maximum(mx.sum(sel).astype(mx.float32), 1.0)
    return cls_loss + loc_loss


def cosine_decay(initial_lr, epoch, total_epochs, min_lr=0.0):
    return min_lr + (initial_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))


def main():
    initial_learning_rate = 1e-2
    total_epochs = 100
    freeze_backbone = True

    data = load_data(DATASET_ROOT, 300)
    anchors = generate_anchors([1, 2, 3, 1 / 2, 1 / 3], feature_map_sizes=[37, 18, 9, 5, 3, 1])
    dataset = prepare_ssd_dataset(data, anchors)
    model = SSD300(num_classes=2)  # pedestrian + background
    # load vgg16 weights
    model.load_weights(
        "/Users/taigaishida/workspace/mlx-models/networks/vgg16/weights.npz",
        strict=False,
    )

    if freeze_backbone:
        for idx, (name, module) in enumerate(model.features.named_modules()[::-1]):
            if idx < 30:
                module.freeze()

    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    trainable_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))

    mx.eval(model.parameters())
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=initial_learning_rate)
    # optimizer = optim.Adam(learning_rate=initial_learning_rate)

    batch_size = 16

    tora = Tora.create_experiment(
        name=f"SSD_{uuid4().hex[:3]}",
        description='finally building a ssd from scratch-ish',
        hyperparams={
            "architecture": "SSD300",
            "batch_size": batch_size,
            "epochs": total_epochs,
            "learning_rate": initial_learning_rate,
            "freeze_backbone": freeze_backbone,
            "num_trainable_params": trainable_params,
            "num_frozen_params": num_params - trainable_params,
        },
        workspace_id="ef58f856-078f-4d28-9808-dd5b522e39bc",
    )
    tora.max_buffer_len = 1

    for epoch in range(total_epochs):
        # current_lr = cosine_decay(initial_learning_rate, epoch, total_epochs, min_learning_rate)
        # optimizer.learning_rate = current_lr

        culm_loss = 0
        num_samples = 0
        for images, loc_targets, cls_targets in dataloader(dataset, batch_size=batch_size):
            loss, grads = loss_and_grad_fn(model, images, loc_targets, cls_targets)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            culm_loss += loss.item() * (images.shape[0])
            num_samples += images.shape[0]

        tora.log('train_loss', step=epoch, value=(culm_loss / num_samples))
        tora.log('lr', step=epoch, value=initial_learning_rate)
        print(f"train loss @{epoch} (lr={initial_learning_rate:.6f})", culm_loss / num_samples)

    image = mx.expand_dims(data[5]["resized_image"], 0)
    pred_loc, pred_cls = model(image)

    detections = decode_predictions(pred_loc, pred_cls, anchors, 9.95, 0.15)
    if detections[0]:
        original_image = np.array(data[5]["resized_image"])
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)

        visualize_detections(
            original_image,
            detections[0],
            class_names=["background", "pedestrian"],
            save_path="detection_visualization.jpg",
        )
        print(f"Saved visualization with {len(detections[0])} detections to detection_visualization.jpg")


if __name__ == "__main__":
    main()
