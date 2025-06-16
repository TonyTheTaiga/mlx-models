from __future__ import annotations
import random
from collections import namedtuple
import json
from pathlib import Path
from dataclasses import dataclass

import cv2
import mlx.core as mx

from networks.ssd.model import SSD300
from networks.ssd.utils import generate_anchors

DATASET_ROOT = Path("/Users/taigaishida/workspace/mlx-models/pedestrians/")


@dataclass
class BBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int


def load_dataset(size=300):
    data = {}

    for img_file in (DATASET_ROOT / "images").iterdir():
        if "mask" in img_file.name:
            continue
        img_id = img_file.stem
        image = cv2.imread(str(img_file))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_file}")
        data[img_id] = {"image": image}

    for ann_file in (DATASET_ROOT / "annotations").iterdir():
        img_id = ann_file.stem
        jsn = json.loads(ann_file.read_text())
        bboxes = [BBox(*anno["bbox"]) for anno in jsn["objects"]]

        if img_id not in data:
            print(f"Warning: annotation for missing image {img_id}")
            continue
        data[img_id]["bboxes"] = bboxes

    for annotation in data.values():
        image, bboxes = annotation["image"], annotation["bboxes"]
        h, w, _ = image.shape
        resized_image = cv2.resize(image, (size, size))
        resized_image = resized_image / 255.0
        resized_image = mx.array(resized_image)

        x_scale = size / w
        y_scale = size / h
        resized_bboxes = [
            [
                bbox.xmin * x_scale,
                bbox.ymin * y_scale,
                bbox.xmax * x_scale,
                bbox.ymax * y_scale,
            ]
            for bbox in bboxes
        ]

        annotation["resized_image"] = resized_image
        annotation["resized_bboxes"] = resized_bboxes

    return data


def split_dataset(data: dict, train_ratio: float = 0.8, seed: int = 42) -> dict[str, list]:
    ids = list(data.keys())
    if seed is not None:
        random.seed(seed)
    random.shuffle(ids)

    split_idx = int(len(ids) * train_ratio)
    train_ids = ids[:split_idx]
    test_ids = ids[split_idx:]

    train = [data[i] for i in train_ids]
    test = [data[i] for i in test_ids]

    return {"train": train, "test": test}


def show_bboxes(dataset: dict):
    """
    Iterate through the dataset and display each image with its bounding boxes overlaid.
    """
    for img_id, item in dataset.items():
        img = item["image"].copy()
        bboxes = item.get("bboxes", [])

        for bbox in bboxes:
            xmin, ymin, xmax, ymax = bbox
            top_left = (int(xmin), int(ymin))
            bottom_right = (int(xmax), int(ymax))
            cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

        # Show the result
        cv2.imshow(f"{img_id}", img)
        key = cv2.waitKey(0)
        # Press 'q' to quit early
        if key == ord("q"):
            break
        cv2.destroyWindow(f"{img_id}")

    cv2.destroyAllWindows()


def dataloader(data, batch_size):
    idx = mx.random.permutation(len(data))
    for start in range(0, len(data), batch_size):
        yield data[idx[start : start + batch_size]]


def main():
    dataset = split_dataset(load_dataset())  # pyright: ignore
    model = SSD300(num_classes=2)  # pedestrian + background
    # load vgg16 weights
    model.load_weights(
        "/Users/taigaishida/workspace/mlx-models/networks/vgg16/weights.npz",
        strict=False,
    )
    image = mx.expand_dims(dataset["train"][0]["resized_image"], 0)
    model(image)
    anchors = generate_anchors([1, 2, 3, 1 / 2, 1 / 3], feature_map_sizes=[37, 18, 9, 5, 3, 1])


if __name__ == "__main__":
    main()
