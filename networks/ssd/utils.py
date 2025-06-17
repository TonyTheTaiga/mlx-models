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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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


def prepare_ssd_dataset(
    data: list[dict], anchors: mx.array, pos_iou_thresh: float = 0.4
) -> tuple[mx.array, mx.array, mx.array]:
    image_batch, loc_targets_batch, cls_targets_batch = [], [], []

    N_priors = anchors.shape[0]
    assert anchors.shape == (N_priors, 4), f"Expected anchors shape ({N_priors}, 4), but got {anchors.shape}"

    # 1) Precompute anchor corners & area
    acx, acy, aw, ah = mx.split(anchors, 4, axis=1)
    axmin = acx - 0.5 * aw
    aymin = acy - 0.5 * ah
    axmax = acx + 0.5 * aw
    aymax = acy + 0.5 * ah
    aarea = aw * ah

    assert acx.shape == (N_priors, 1)
    assert aymin.shape == (N_priors, 1)
    assert aarea.shape == (N_priors, 1)

    for item in data:
        img = item["resized_image"]
        H, W, C = img.shape
        assert img.ndim == 3, f"Expected image to be 3D (C, H, W), but got {img.ndim}D"

        # 2) Load & normalize GT boxes
        coords = mx.array(item["resized_bboxes"], dtype=mx.float32)
        M = coords.shape[0]

        if M == 0:  # maybe this should be a error
            loc_targets = mx.zeros_like(anchors)
            cls_targets = mx.zeros((N_priors,), dtype=mx.int32)
            assert loc_targets.shape == (N_priors, 4)
            assert cls_targets.shape == (N_priors,)
        else:
            labels = mx.ones((M,), dtype=mx.int32)
            assert labels.shape == (M,)

            x1, y1, x2, y2 = mx.split(coords, 4, axis=1)
            assert x1.shape == (M, 1)

            gxmin, gymin, gxmax, gymax = x1 / W, y1 / H, x2 / W, y2 / H
            assert gxmin.shape == (M, 1)

            gcx, gcy = (gxmin + gxmax) * 0.5, (gymin + gymax) * 0.5
            gw, gh = gxmax - gxmin, gymax - gymin
            gt_centers = mx.concat([gcx, gcy, gw, gh], axis=1)
            garea = gw * gh
            assert gt_centers.shape == (M, 4)
            assert garea.shape == (M, 1)

            # 3) Pairwise IoU
            gx0, gy0, gx1, gy1 = gxmin.T, gymin.T, gxmax.T, gymax.T
            assert gx0.shape == (1, M)

            ix0, iy0 = mx.maximum(axmin, gx0), mx.maximum(aymin, gy0)
            ix1, iy1 = mx.minimum(axmax, gx1), mx.minimum(aymax, gy1)
            assert ix0.shape == (N_priors, M)

            iw, ih = mx.clip(ix1 - ix0, 0, None), mx.clip(iy1 - iy0, 0, None)
            inter = iw * ih
            union = aarea + garea.T - inter
            iou = inter / union
            assert iou.shape == (N_priors, M)

            # 4) Match priors → GTs
            best_iou = mx.max(iou, axis=1)
            best_idx = mx.argmax(iou, axis=1)
            forced = mx.argmax(iou, axis=0)
            assert best_iou.shape == (N_priors,)
            assert best_idx.shape == (N_priors,)
            assert forced.shape == (M,)

            pos_mask = best_iou >= pos_iou_thresh
            pos_mask[forced] = True
            assert pos_mask.shape == (N_priors,)

            # 5) Classification targets
            matched_labels = labels[best_idx]
            cls_targets = mx.where(pos_mask, matched_labels, mx.zeros_like(matched_labels))
            assert matched_labels.shape == (N_priors,)
            assert cls_targets.shape == (N_priors,)

            # 6) Localization targets
            matched_gt = gt_centers[best_idx]
            t_xy = (matched_gt[:, :2] - anchors[:, :2]) / anchors[:, 2:]
            t_wh = mx.log(matched_gt[:, 2:] / anchors[:, 2:])
            offsets = mx.concat([t_xy, t_wh], axis=1)
            assert matched_gt.shape == (N_priors, 4)
            assert t_xy.shape == (N_priors, 2)
            assert t_wh.shape == (N_priors, 2)
            assert offsets.shape == (N_priors, 4)

            pos_mask_exp = mx.expand_dims(pos_mask, 1)
            loc_targets = mx.where(pos_mask_exp, offsets, mx.zeros_like(offsets))
            assert pos_mask_exp.shape == (N_priors, 1)
            assert loc_targets.shape == (N_priors, 4)

        image_batch.append(img)
        loc_targets_batch.append(loc_targets)
        cls_targets_batch.append(cls_targets)

    num_samples = len(data)
    final_images = mx.stack(image_batch, axis=0)
    final_locs = mx.stack(loc_targets_batch, axis=0)
    final_clss = mx.stack(cls_targets_batch, axis=0)

    # Assert final batch shapes
    _, C, H, W = final_images.shape
    assert final_images.shape == (num_samples, C, H, W), (
        f"Expected final_images shape ({num_samples}, {C}, {H}, {W}), but got {final_images.shape}"
    )
    assert final_locs.shape == (num_samples, N_priors, 4), (
        f"Expected final_locs shape ({num_samples}, {N_priors}, 4), but got {final_locs.shape}"
    )
    assert final_clss.shape == (num_samples, N_priors), (
        f"Expected final_clss shape ({num_samples}, {N_priors}), but got {final_clss.shape}"
    )

    return final_images, final_locs, final_clss


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
    delta = (max_scale - min_scale) / (m - 1)
    return [min_scale + delta * (k - 1) for k in range(1, steps + 1)]


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
