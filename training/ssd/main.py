from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
from mlx.utils import tree_flatten
from tora import Tora
from utils import decode_predictions, visualize_detections

from networks.ssd.model import SSD300
from networks.ssd.utils import generate_anchors, load_data, prepare_ssd_dataset

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_yaml_config(path: Path) -> dict[str, Any]:
    with path.expanduser().open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def resolve_path(value: str | None, base: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base / path).resolve()
    return path


def build_config(config_path: Path = CONFIG_PATH) -> SimpleNamespace:
    cfg = load_yaml_config(config_path)
    base = config_path.parent
    dataset_root = resolve_path(cfg.get("dataset_root"), base) or Path(
        "/Users/taigaishida/workspace/mlx-models/pedestrians/"
    )

    def to_float_list(values, default):
        seq = values if values is not None else default
        return [float(v) for v in seq]

    def to_int_list(values, default):
        seq = values if values is not None else default
        return [int(v) for v in seq]

    return SimpleNamespace(
        dataset_root=dataset_root,
        input_size=int(cfg.get("input_size", 300)),
        batch_size=int(cfg.get("batch_size", 16)),
        initial_lr=float(cfg.get("initial_learning_rate", 2e-2)),
        total_epochs=int(cfg.get("total_epochs", 250)),
        optimizer=cfg.get("optimizer", "SGD"),
        freeze_backbone=bool(cfg.get("freeze_backbone", True)),
        load_pretrained_weights=bool(cfg.get("load_pretrained_weights", False)),
        pretrained_weights=resolve_path(cfg.get("pretrained_weights_path"), base),
        anchor_aspect_ratios=to_float_list(
            cfg.get("anchor_aspect_ratios"), [1.0, 2.0, 3.0, 0.5, 1 / 3]
        ),
        feature_map_sizes=to_int_list(cfg.get("feature_map_sizes"), [37, 18, 9, 5, 3, 1]),
        conf_threshold=float(cfg.get("confidence_threshold", 0.9)),
        nms_threshold=float(cfg.get("nms_threshold", 0.15)),
        description=cfg.get(
            "description", "SSD300, VGG16 backbone, pretrained weights obtained via torchvision"
        ),
        visualization_path=resolve_path(cfg.get("visualization_path"), base)
        or (base / "detection_visualization.jpg"),
    )


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
    loc_loss_all = nn.losses.huber_loss(loc_preds, loc_targets, reduction="none")
    loc_loss = mx.sum(mx.where(mx.expand_dims(pos, -1), loc_loss_all, 0.0))
    loc_loss = alpha * loc_loss / mx.maximum(mx.sum(pos).astype(mx.float32), 1.0)
    cls_loss_all = nn.losses.cross_entropy(
        cls_preds.reshape(-1, C), cls_targets.reshape(-1), reduction="none"
    ).reshape(B, N)

    # detach
    scores = mx.stop_gradient(cls_loss_all * neg)

    ranked = mx.argsort(scores, axis=1)
    k = mx.minimum(3 * mx.sum(pos, axis=1), mx.sum(neg, axis=1))
    arange_N = mx.arange(N)

    hard_neg = arange_N[None, :] >= (N - k[:, None])
    hard_neg = mx.take_along_axis(hard_neg, ranked, axis=1)

    sel = pos | hard_neg
    cls_loss = mx.sum(cls_loss_all * sel) / mx.maximum(mx.sum(sel).astype(mx.float32), 1.0)
    return (cls_loss + loc_loss, cls_loss, loc_loss)


def cosine_decay(initial_lr, epoch, total_epochs, min_lr=0.0):
    return min_lr + (initial_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))


def main(cfg: SimpleNamespace | None = None):
    cfg = cfg or build_config()
    cfg.visualization_path.parent.mkdir(parents=True, exist_ok=True)

    initial_learning_rate = cfg.initial_lr
    total_epochs = cfg.total_epochs
    freeze_backbone = cfg.freeze_backbone
    load_pretrained_weights = cfg.load_pretrained_weights
    optim_type = cfg.optimizer
    batch_size = cfg.batch_size

    data = load_data(cfg.dataset_root, cfg.input_size)
    anchors = generate_anchors(cfg.anchor_aspect_ratios, feature_map_sizes=cfg.feature_map_sizes)
    dataset = prepare_ssd_dataset(data, anchors)
    model = SSD300(num_classes=2)  # pedestrian + background

    if load_pretrained_weights:
        if cfg.pretrained_weights is None:
            raise ValueError(
                "pretrained weights path must be provided when load_pretrained_weights is True"
            )
        model.load_weights(str(cfg.pretrained_weights), strict=False)

    if freeze_backbone:
        if not load_pretrained_weights:
            print("load_pretrained_weights set to False, freezing backbone is not supported")
            freeze_backbone = False
        else:
            for idx, (name, module) in enumerate(model.features.named_modules()[::-1]):
                if idx < 30:
                    module.freeze()

    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    trainable_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))

    mx.eval(model.parameters())
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    if optim_type == "SGD":
        optimizer = optim.SGD(learning_rate=initial_learning_rate)
    else:
        optimizer = optim.Adam(learning_rate=initial_learning_rate)

    tora = Tora.create_experiment(
        name=f"SSD_{uuid4().hex[:3]}",
        description=cfg.description,
        hyperparams={
            "architecture": "SSD300",
            "batch_size": batch_size,
            "epochs": total_epochs,
            "learning_rate": initial_learning_rate,
            "optimizer": optim_type,
            "freeze_backbone": freeze_backbone,
            "num_trainable_params": trainable_params,
            "num_frozen_params": num_params - trainable_params,
        },
    )
    tora.max_buffer_len = 1

    for epoch in range(total_epochs):
        # current_lr = cosine_decay(initial_learning_rate, epoch, total_epochs, min_learning_rate)
        # optimizer.learning_rate = current_lr

        culm_loss = 0
        culm_loc_loss = 0
        culm_cls_loss = 0
        num_samples = 0
        for images, loc_targets, cls_targets in dataloader(dataset, batch_size=batch_size):
            (loss, cls_loss, loc_loss), grads = loss_and_grad_fn(
                model, images, loc_targets, cls_targets
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            culm_loss += loss.item() * (images.shape[0])
            culm_loc_loss += loc_loss.item() * (images.shape[0])
            culm_cls_loss += cls_loss.item() * (images.shape[0])

            num_samples += images.shape[0]

        tora.log("train_loss", step=epoch, value=(culm_loss / num_samples))
        tora.log("train_cls_loss", step=epoch, value=(culm_cls_loss / num_samples))
        tora.log("train_loc_loss", step=epoch, value=(culm_loc_loss / num_samples))
        tora.log("lr", step=epoch, value=initial_learning_rate)
        print(f"train loss @{epoch} (lr={initial_learning_rate:.6f})", culm_loss / num_samples)

    image = mx.expand_dims(data[5]["resized_image"], 0)
    pred_loc, pred_cls = model(image)

    detections = decode_predictions(
        pred_loc, pred_cls, anchors, cfg.conf_threshold, cfg.nms_threshold
    )
    if detections[0]:
        original_image = np.array(data[5]["image"])
        if original_image.max() <= 1.0:
            original_image = (original_image * 255).astype(np.uint8)

        visualize_detections(
            original_image,
            detections[0],
            class_names=["background", "pedestrian"],
            save_path=str(cfg.visualization_path),
        )
        print(
            f"Saved visualization with {len(detections[0])} detections to {cfg.visualization_path}"
        )


if __name__ == "__main__":
    main()
