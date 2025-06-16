from __future__ import annotations
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import math

from networks.ssd.model import SSD300
from networks.ssd.utils import generate_anchors, load_data, prepare_ssd_dataset
from utils import decode_predictions, visualize_detections

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
    """
    Calculates the SSD loss, incorporating hard negative mining for classification
    and applying localization loss only to positive samples, using MLX-friendly operations.

    Args:
        model: The SSD model.
        images: Batch of input images (num_samples, C, H, W).
        loc_targets: Ground truth localization targets (num_samples, N_priors, 4).
        cls_targets: Ground truth class targets (num_samples, N_priors).
        alpha: Weighting factor for the localization loss.

    Returns:
        A scalar MLX array representing the total combined loss.
    """
    loc_preds, cls_preds = model(images)
    # loc_preds: (B, N_priors, 4) - Predicted bounding box offsets
    # cls_preds: (B, N_priors, num_classes) - Predicted class logits/scores

    batch_size, N_priors, num_classes = cls_preds.shape

    # --- Masks ---
    # Create boolean masks for positive (object) and negative (background) anchors
    pos_mask = cls_targets > 0  # Shape: (B, N_priors) - True where there's an object
    neg_mask = cls_targets == 0 # Shape: (B, N_priors) - True where it's background

    # Flatten relevant tensors for operations that process all anchors in a batch
    cls_preds_flat = cls_preds.reshape(-1, num_classes) # (B * N_priors, num_classes)
    cls_targets_flat = cls_targets.reshape(-1)         # (B * N_priors,)
    pos_mask_flat = pos_mask.reshape(-1)               # (B * N_priors,)
    neg_mask_flat = neg_mask.reshape(-1)               # (B * N_priors,)


    # --- 1. Localization Loss (L_loc) ---
    # Goal: Calculate Smooth L1 loss ONLY for positive anchors.

    # Expand pos_mask to (B, N_priors, 1) so it can broadcast with (B, N_priors, 4) for `mx.where`
    pos_mask_expanded_for_loc = mx.expand_dims(pos_mask, axis=-1)

    # Calculate element-wise Huber (Smooth L1) loss for ALL anchors.
    all_huber_losses = nn.losses.huber_loss(loc_preds, loc_targets, reduction='none') # Shape: (B, N_priors, 4)

    # Use `mx.where` to "mask out" (set to 0.0) the losses for non-positive anchors.
    masked_loc_losses = mx.where(pos_mask_expanded_for_loc, all_huber_losses, mx.array(0.0))

    # Sum all masked losses. Only positive losses contribute to the sum.
    sum_loc_losses = mx.sum(masked_loc_losses)

    # Count the total number of positive anchors across the batch for normalization.
    num_pos_anchors = mx.sum(pos_mask).astype(mx.float32)

    # Calculate the mean localization loss. `mx.maximum(1e-6, ...)` prevents division by zero if `num_pos_anchors` is 0.
    loc_loss = sum_loc_losses / mx.maximum(mx.array(1e-6), num_pos_anchors)


    # --- 2. Confidence/Classification Loss (L_conf) with Hard Negative Mining ---
    # Goal: Calculate Cross-Entropy loss for positive anchors and a subset of "hard" negative anchors.

    # Calculate element-wise Cross-Entropy loss for ALL anchors (both positive and negative/background).
    all_cls_losses = nn.losses.cross_entropy(cls_preds_flat, cls_targets_flat, reduction="none") # (B * N_priors,)

    # --- Handle Positive Classification Losses ---
    # Sum losses for all positive samples.
    sum_pos_cls_losses = mx.sum(mx.where(pos_mask_flat, all_cls_losses, mx.array(0.0)))

    # Get the count of actual positive samples in the current batch.
    num_actual_pos_samples = mx.sum(pos_mask_flat)

    # --- Hard Negative Mining for Negative (Background) Samples ---
    # Determine how many hard negative samples to select (3x positives, capped by total negatives).
    num_hard_neg_to_select = mx.minimum(3 * num_actual_pos_samples, mx.sum(neg_mask_flat))
    num_hard_neg_to_select_py = num_hard_neg_to_select.item() # Convert MLX scalar to Python int

    # Initialize hard negative losses sum
    sum_hard_neg_cls_losses = mx.array(0.0)
    num_hard_neg_selected = 0

    if num_hard_neg_to_select_py > 0:
        # Get only negative losses (mask out positive ones with 0)
        neg_cls_losses_only = mx.where(neg_mask_flat, all_cls_losses, mx.array(0.0))

        # Sort in descending order to get hardest negatives first
        sorted_neg_cls_losses = mx.sort(neg_cls_losses_only)[::-1]

        # Select the top hard negatives (highest losses)
        hard_neg_cls_losses_selected = sorted_neg_cls_losses[:num_hard_neg_to_select_py]

        # Use mx.where to filter out zeros instead of boolean indexing
        non_zero_mask = hard_neg_cls_losses_selected > 0
        hard_neg_cls_losses_selected = mx.where(non_zero_mask, hard_neg_cls_losses_selected, mx.array(0.0))

        sum_hard_neg_cls_losses = mx.sum(hard_neg_cls_losses_selected)
        num_hard_neg_selected = mx.sum(non_zero_mask.astype(mx.int32))

    # Combine the summed positive and hard negative losses.
    combined_losses_sum = sum_pos_cls_losses + sum_hard_neg_cls_losses

    # Calculate the total number of samples contributing to this combined classification loss mean.
    total_samples_for_cls_loss = num_actual_pos_samples + num_hard_neg_selected

    # Calculate the mean classification loss, avoiding division by zero.
    cls_loss = combined_losses_sum / mx.maximum(mx.array(1e-6), total_samples_for_cls_loss.astype(mx.float32))

    # --- Total Combined Loss ---
    total_loss = cls_loss + alpha * loc_loss

    print(cls_loss.item(), loc_loss.item())
    return total_loss

def cosine_decay(initial_lr, epoch, total_epochs, min_lr=0.0):
    return min_lr + (initial_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))

def main():
    initial_learning_rate = 1e-4
    total_epochs = 50
    min_learning_rate = 1e-5

    data = load_data(DATASET_ROOT, 300)
    anchors = generate_anchors([1, 2, 3, 1 / 2, 1 / 3], feature_map_sizes=[37, 18, 9, 5, 3, 1])
    dataset = prepare_ssd_dataset(data, anchors)
    model = SSD300(num_classes=2)  # pedestrian + background
    # load vgg16 weights
    # model.load_weights(
    #     "/Users/taigaishida/workspace/mlx-models/networks/vgg16/weights.npz",
    #     strict=False,
    # )
    # for idx, (name, module) in enumerate(model.features.named_modules()[::-1]):
    #     if idx < 30:
    #         module.freeze()

    mx.eval(model.parameters())
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=initial_learning_rate)

    image = mx.expand_dims(data[0]["resized_image"], 0)
    pred_loc, pred_cls = model(image)

    for epoch in range(total_epochs):
        # Update learning rate with cosine decay
        current_lr = cosine_decay(initial_learning_rate, epoch, total_epochs, min_learning_rate)
        optimizer.learning_rate = current_lr

        culm_loss = 0
        num_samples = 0
        for images, loc_targets, cls_targets in dataloader(dataset, batch_size=8):
            loss, grads = loss_and_grad_fn(model, images, loc_targets, cls_targets)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            culm_loss += loss.item() * (images.shape[0])
            num_samples += images.shape[0]

        print(f"train loss @{epoch} (lr={current_lr:.6f})", culm_loss / num_samples)


if __name__ == "__main__":
    main()
