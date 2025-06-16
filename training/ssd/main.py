from __future__ import annotations
from pathlib import Path

import mlx.core as mx

from networks.ssd.model import SSD300
from networks.ssd.utils import generate_anchors, load_data, prepare_ssd_dataset

DATASET_ROOT = Path("/Users/taigaishida/workspace/mlx-models/pedestrians/")


def dataloader(data, batch_size):
    idx = mx.random.permutation(len(data))
    for start in range(0, len(data), batch_size):
        yield data[idx[start : start + batch_size]]


def main():
    data = load_data(DATASET_ROOT, 300)
    anchors = generate_anchors([1, 2, 3, 1 / 2, 1 / 3], feature_map_sizes=[37, 18, 9, 5, 3, 1])
    dataset = prepare_ssd_dataset(data, anchors)
    model = SSD300(num_classes=2)  # pedestrian + background
    # load vgg16 weights
    model.load_weights(
        "/Users/taigaishida/workspace/mlx-models/networks/vgg16/weights.npz",
        strict=False,
    )
    image = mx.expand_dims(data[0]["resized_image"], 0)
    model(image)


if __name__ == "__main__":
    main()
