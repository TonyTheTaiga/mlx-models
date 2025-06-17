from __future__ import annotations

import mlx.core as mx
import cv2
import numpy as np


def decode_predictions(loc_preds, cls_preds, anchors, conf_thresh=0.5, nms_thresh=0.4):
    batch_size = loc_preds.shape[0]
    batch_detections = []

    for b in range(batch_size):
        loc_pred = loc_preds[b]
        cls_pred = cls_preds[b]
        cls_prob = mx.softmax(cls_pred, axis=-1)
        if cls_prob.shape[-1] > 1:
            max_conf = mx.max(cls_prob[:, 1:], axis=-1)
            class_ids = mx.argmax(cls_prob[:, 1:], axis=-1) + 1  # Add 1 to skip background
        else:
            max_conf = cls_prob[:, 0]
            class_ids = mx.zeros_like(max_conf, dtype=mx.int32)

        conf_mask = max_conf > conf_thresh
        if mx.sum(conf_mask) == 0:
            batch_detections.append([])
            continue

        conf_mask_np = np.array(conf_mask)
        loc_pred_np = np.array(loc_pred)
        anchors_np = np.array(anchors)
        max_conf_np = np.array(max_conf)
        class_ids_np = np.array(class_ids)

        filtered_locs_np = loc_pred_np[conf_mask_np]
        filtered_anchors_np = anchors_np[conf_mask_np]
        filtered_confs_np = max_conf_np[conf_mask_np]
        filtered_classes_np = class_ids_np[conf_mask_np]

        filtered_locs = mx.array(filtered_locs_np)
        filtered_anchors = mx.array(filtered_anchors_np)
        filtered_confs = mx.array(filtered_confs_np)
        filtered_classes = mx.array(filtered_classes_np)

        decoded_cx = filtered_locs[:, 0] * filtered_anchors[:, 2] + filtered_anchors[:, 0]
        decoded_cy = filtered_locs[:, 1] * filtered_anchors[:, 3] + filtered_anchors[:, 1]
        decoded_w = mx.exp(filtered_locs[:, 2]) * filtered_anchors[:, 2]
        decoded_h = mx.exp(filtered_locs[:, 3]) * filtered_anchors[:, 3]

        x1 = decoded_cx - decoded_w * 0.5
        y1 = decoded_cy - decoded_h * 0.5
        x2 = decoded_cx + decoded_w * 0.5
        y2 = decoded_cy + decoded_h * 0.5

        x1 = mx.clip(x1, 0, 1)
        y1 = mx.clip(y1, 0, 1)
        x2 = mx.clip(x2, 0, 1)
        y2 = mx.clip(y2, 0, 1)

        detections = mx.stack([x1, y1, x2, y2], axis=1)
        detections_np = np.array(detections)
        scores_np = np.array(filtered_confs)
        classes_np = np.array(filtered_classes)
        keep_indices = []
        sorted_indices = np.argsort(scores_np)[::-1]

        while len(sorted_indices) > 0:
            current = sorted_indices[0]
            keep_indices.append(current)

            if len(sorted_indices) == 1:
                break

            current_box = detections_np[current]
            remaining_boxes = detections_np[sorted_indices[1:]]

            # Calculate IoU
            x1_inter = np.maximum(current_box[0], remaining_boxes[:, 0])
            y1_inter = np.maximum(current_box[1], remaining_boxes[:, 1])
            x2_inter = np.minimum(current_box[2], remaining_boxes[:, 2])
            y2_inter = np.minimum(current_box[3], remaining_boxes[:, 3])

            inter_area = np.maximum(0, x2_inter - x1_inter) * np.maximum(0, y2_inter - y1_inter)

            current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
            remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (
                remaining_boxes[:, 3] - remaining_boxes[:, 1]
            )

            union_area = current_area + remaining_areas - inter_area
            iou = np.divide(inter_area, union_area, out=np.zeros_like(inter_area), where=union_area != 0)

            keep_mask = iou < nms_thresh
            sorted_indices = sorted_indices[1:][keep_mask]

        final_detections = []
        for idx in keep_indices:
            final_detections.append(
                {"bbox": detections_np[idx], "confidence": float(scores_np[idx]), "class_id": int(classes_np[idx])}
            )

        batch_detections.append(final_detections)

    return batch_detections


def visualize_detections(image, detections, class_names=None, save_path=None):
    if image.dtype == np.float32 or image.dtype == np.float64:
        vis_image = (image * 255).astype(np.uint8)
    else:
        vis_image = image.copy()

    h, w = vis_image.shape[:2]
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    for det in detections:
        bbox = det["bbox"]  # [x1, y1, x2, y2] in normalized coords
        confidence = det["confidence"]
        class_id = det["class_id"]

        # Convert to pixel coordinates
        x1 = int(bbox[0] * w)
        y1 = int(bbox[1] * h)
        x2 = int(bbox[2] * w)
        y2 = int(bbox[3] * h)

        color = colors[class_id % len(colors)]
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        if class_names and class_id < len(class_names):
            label = f"{class_names[class_id]}: {confidence:.2f}"
        else:
            label = f"Class {class_id}: {confidence:.2f}"

        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(vis_image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(vis_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

    return vis_image
