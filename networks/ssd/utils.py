from collections import namedtuple
import random
import json
from pathlib import Path
import math

import mlx.core as mx
import cv2


def load_data(dataset_root: Path, size: int = 300):
    data = {}

    for img_file in (dataset_root / "images").iterdir():
        if "mask" in img_file.name:
            continue
        img_id = img_file.stem
        image = cv2.imread(str(img_file))
        if image is None:
            raise FileNotFoundError(f"Could not load image: {img_file}")
        data[img_id] = {"image": image}

    for ann_file in (dataset_root / "annotations").iterdir():
        img_id = ann_file.stem
        jsn = json.loads(ann_file.read_text())
        bboxes = [anno["bbox"] for anno in jsn["objects"]]

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
                bbox[0] * x_scale,
                bbox[1] * y_scale,
                bbox[2] * x_scale,
                bbox[3] * y_scale,
            ]
            for bbox in bboxes
        ]

        annotation["resized_image"] = resized_image
        annotation["resized_bboxes"] = resized_bboxes

    return list(data.values())


def prepare_ssd_dataset(data: list[dict], anchors: mx.array, pos_iou_thresh: float = 0.5) -> list[tuple]:
    outputs = []
    N_priors = anchors.shape[0]
    assert anchors.shape == (N_priors, 4), f"Expected anchors shape ({N_priors}, 4), but got {anchors.shape}"

    # 1) Precompute anchor corners & area
    acx, acy, aw, ah = mx.split(anchors, 4, axis=1)
    axmin = acx - 0.5 * aw
    aymin = acy - 0.5 * ah
    axmax = acx + 0.5 * aw
    aymax = acy + 0.5 * ah
    aarea = aw * ah  # (N_priors, 1)
    assert aarea.shape == (N_priors, 1), f"Expected aarea shape ({N_priors}, 1), but got {aarea.shape}"

    for item in data:
        img = item["resized_image"]  # (C, H, W)
        _, H, W = img.shape

        # 2) Load & normalize GT boxes; all are pedestrians (class=1)
        coords = mx.array(item["bboxes"], dtype=mx.float32)  # (M,4)
        M = coords.shape[0]

        if M == 0:  # Handle cases with no ground truth boxes
            loc_targets = mx.zeros_like(anchors)
            cls_targets = mx.zeros((anchors.shape[0],), dtype=mx.int32)
            Sample = namedtuple("Sample", ["image", "loc_targets", "cls_targets"])
            outputs.append(Sample(img, loc_targets, cls_targets))
            continue

        assert coords.shape == (M, 4), f"Expected GT coords shape ({M}, 4), but got {coords.shape}"
        labels = mx.ones((M,), dtype=mx.int32)  # (M,)

        x1, y1, x2, y2 = mx.split(coords, 4, axis=1)
        gxmin = x1 / W
        gymin = y1 / H
        gxmax = x2 / W
        gymax = y2 / H

        # center‐form
        gcx = (gxmin + gxmax) * 0.5
        gcy = (gymin + gymax) * 0.5
        gw = gxmax - gxmin
        gh = gymax - gymin
        gt_centers = mx.concat([gcx, gcy, gw, gh], axis=1)  # (M,4)
        garea = gw * gh  # (M,1)
        assert gt_centers.shape == (M, 4), f"Expected gt_centers shape ({M}, 4), but got {gt_centers.shape}"

        # 3) Pairwise IoU: anchors (N,1) vs GT (1,M) -> (N,M)
        gx0, gy0, gx1, gy1 = gxmin.T, gymin.T, gxmax.T, gymax.T

        ix0 = mx.maximum(axmin, gx0)
        iy0 = mx.maximum(aymin, gy0)
        ix1 = mx.minimum(axmax, gx1)
        iy1 = mx.minimum(aymax, gy1)

        iw = mx.clip(ix1 - ix0, 0, None)
        ih = mx.clip(iy1 - iy0, 0, None)
        inter = iw * ih

        union = aarea + garea.T - inter
        iou = inter / union  # (N_priors, M)
        assert iou.shape == (N_priors, M), f"Expected iou shape ({N_priors}, {M}), but got {iou.shape}"

        # 4) Match priors → GTs
        best_iou = mx.max(iou, axis=1)
        best_idx = mx.argmax(iou, axis=1)
        assert best_iou.shape == (N_priors,), f"Expected best_iou shape ({N_priors},), but got {best_iou.shape}"

        forced = mx.argmax(iou, axis=0)
        pos_mask = best_iou >= pos_iou_thresh
        pos_mask[forced] = True
        assert pos_mask.shape == (N_priors,), f"Expected pos_mask shape ({N_priors},), but got {pos_mask.shape}"

        # 5) Classification targets (0=bg, 1=pedestrian)
        matched_labels = labels[best_idx]
        cls_targets = mx.where(pos_mask, matched_labels, mx.zeros_like(matched_labels))
        assert cls_targets.shape == (N_priors,), (
            f"Expected cls_targets shape ({N_priors},), but got {cls_targets.shape}"
        )

        # 6) Localization targets
        matched_gt = gt_centers[best_idx]
        t_xy = (matched_gt[:, :2] - anchors[:, :2]) / anchors[:, 2:]
        t_wh = mx.log(matched_gt[:, 2:] / anchors[:, 2:])
        offsets = mx.concat([t_xy, t_wh], axis=1)
        assert offsets.shape == (N_priors, 4), f"Expected offsets shape ({N_priors}, 4), but got {offsets.shape}"

        pos_mask_exp = mx.expand_dims(pos_mask, 1)
        loc_targets = mx.where(pos_mask_exp, offsets, mx.zeros_like(offsets))
        assert loc_targets.shape == (N_priors, 4), (
            f"Expected loc_targets shape ({N_priors}, 4), but got {loc_targets.shape}"
        )

        Sample = namedtuple("Sample", ["image", "loc_targets", "cls_targets"])
        outputs.append(Sample(img, loc_targets, cls_targets))

    return outputs


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


def linterpolate(min_scale, max_scale, m, steps):
    Δ = (max_scale - min_scale) / (m - 1)
    return [min_scale + Δ * (k - 1) for k in range(1, steps + 1)]


def generate_anchors(ratios: list[float], feature_map_sizes: list[int]):
    scales = linterpolate(0.2, 0.9, len(feature_map_sizes), len(feature_map_sizes) + 1)
    priors_all = []
    for feature_map_index, K in enumerate(feature_map_sizes):
        whs = []
        scale = scales[feature_map_index]
        for ratio in ratios:
            if (ratio == 3 or ratio == 1 / 3) and feature_map_index in [0, 4, 5]:
                continue

            whs.append((scale * math.sqrt(ratio), scale / math.sqrt(ratio)))

        whs.append(
            (
                math.sqrt(scale * scales[feature_map_index + 1]),
                math.sqrt(scale * scales[feature_map_index + 1]),
            )
        )

        for i in range(K):
            for j in range(K):
                pos_i, pos_j = (i + 0.5) / K, (j + 0.5) / K
                priors_all.extend([(pos_i, pos_j, w, h) for (w, h) in whs])

    anchors = mx.array(priors_all)
    return anchors


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
