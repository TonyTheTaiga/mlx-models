from __future__ import annotations

import cv2
import math
from pathlib import Path
from uuid import uuid4
import numpy as np

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from networks.image_gs import build_model
from networks.image_gs.util import image_gradient_map, build_prob_map
from tora import Tora


def build_probability_mass_from_error_map(error_map: mx.array) -> np.ndarray:
    error_numpy = np.array(error_map, dtype=np.float64)
    probabilities = np.nan_to_num(error_numpy, nan=0.0, posinf=0.0, neginf=0.0).reshape(-1)
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0.0:
        size = probabilities.size
        return np.full(size, 1.0 / size, dtype=np.float64)
    probabilities = probabilities / total
    if probabilities.size > 1:
        probabilities[-1] = max(0.0, 1.0 - probabilities[:-1].sum())
    total_final = probabilities.sum()
    if not np.isfinite(total_final) or total_final <= 0.0:
        size = probabilities.size
        return np.full(size, 1.0 / size, dtype=np.float64)
    return probabilities / total_final


def sampled_l1_loss(model, target_image: mx.array, sample_mask: mx.array, num_samples: int = 10_000) -> mx.array:
    rendered = model(target_image)
    rendered = rendered.astype(mx.float32)
    target_image = target_image.astype(mx.float32)
    pixelwise_absolute_error = mx.abs(rendered - target_image)
    sampled_sum = mx.sum(pixelwise_absolute_error * sample_mask)
    loss_value = sampled_sum / mx.array(float(max(1, num_samples)), dtype=mx.float32)
    return loss_value


def compute_psnr(x: mx.array, y: mx.array, max_val: float = 1.0) -> float:
    mse = mx.mean((x.astype(mx.float32) - y.astype(mx.float32)) ** 2).item()
    if mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(max_val) - 10.0 * math.log10(mse))


def main(image_path: str,
         epochs: int = 2,
         lr: float = 1e-2,
         num_init: int = 2000,
         tile: int = 16,
         num_samples: int = 10_000,
         sample_mix: float = 0.5,
         seed: int | None = 42,
         workspace_id: str | None = None,
         save: Path | None = None,
         budget: int | None = None,
         add_every: int = 500):
    image_bgr = cv2.imread(image_path, flags=cv2.IMREAD_COLOR_RGB)
    if image_bgr is None:
        raise RuntimeError("failed to load image")

    target_image = mx.array(image_bgr).astype(mx.float32) / 255.0
    model = build_model(target_image, num_initial_gaussians=num_init, tile_size=tile)
    mx.eval(model.parameters())

    model(target_image)

    gradient_map = image_gradient_map(target_image)
    probability_mass_mx = build_prob_map(gradient_map, sample_mix).reshape(-1)
    gradient_probabilities = np.array(probability_mass_mx)
    height, width, _ = image_bgr.shape
    random_generator = np.random.default_rng(seed)

    loss_and_gradient_fn = nn.value_and_grad(model, sampled_l1_loss)
    optimizer = optim.Adam(learning_rate=lr)

    total_budget = int(budget) if budget is not None else int(num_init)
    total_budget = max(total_budget, int(num_init))
    if total_budget > model.num_gaussians:
        last_target_epoch = epochs - add_every
        if last_target_epoch < 1:
            last_target_epoch = 1
            print(f"[densify] warning: add_every (={add_every}) >= epochs (={epochs}); target at epoch 1")
        multiples = [k * add_every for k in range(1, max(1, last_target_epoch // add_every) + 1)]
        if multiples and multiples[-1] == last_target_epoch:
            add_epochs = multiples
        else:
            add_epochs = multiples + ([last_target_epoch] if last_target_epoch not in multiples else [])
        add_epochs = sorted(set(e for e in add_epochs if 1 <= e <= last_target_epoch))
        approx_per_step = (total_budget - num_init) / max(1, len(add_epochs))
        print(
            f"[densify] enabled: add_every={add_every}, last_target_epoch={last_target_epoch}, "
            f"steps={len(add_epochs)}, target_budget={total_budget}, ~{approx_per_step:.1f}/step"
        )
    else:
        print(
            f"[densify] disabled: budget={total_budget} <= initial={model.num_gaussians}. "
            f"Pass --budget larger than --num_init to enable."
        )

    num_params = sum(v.size for _, v in tree_flatten(model.parameters()))
    trainable_params = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    exp_name = f"ImageGS_{uuid4().hex[:3]}"
    tora_kwargs = {}
    if workspace_id:
        tora_kwargs["workspace_id"] = workspace_id
    tora = Tora.create_experiment(
        name=exp_name,
        description="Image Gaussian Splatting: single-image reconstruction",
        hyperparams={
            "epochs": epochs,
            "learning_rate": lr,
            "num_initial_gaussians": num_init,
            "tile_size": tile,
            "num_samples": num_samples,
            "sample_mix": sample_mix,
            "num_params": num_params,
            "num_trainable_params": trainable_params,
        },
        **tora_kwargs,
    )
    tora.max_buffer_len = 1

    print(f"initial_gaussians={model.num_gaussians}")
    tora.metric(name="num_gaussians", value=float(model.num_gaussians), step_or_epoch=0)

    for epoch in range(epochs):
        num_sampled_pixels = min(int(num_samples), height * width)
        sampled_indices = random_generator.choice(height * width, size=num_sampled_pixels, replace=False, p=gradient_probabilities)
        sampled_rows = (sampled_indices // width).astype(np.int32)
        sampled_cols = (sampled_indices % width).astype(np.int32)
        sample_mask_numpy = np.zeros((height, width, 1), dtype=np.float32)
        sample_mask_numpy[sampled_rows, sampled_cols, 0] = 1.0
        sample_mask_mx = mx.array(sample_mask_numpy)

        loss_value, gradients = loss_and_gradient_fn(model, target_image, sample_mask_mx, num_sampled_pixels)
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)

        rendered = model(target_image)
        current_psnr = compute_psnr(rendered, target_image)
        print(f"epoch {epoch+1}/{epochs}  loss={loss_value.item():.6f}  psnr={current_psnr:.2f} dB  gaussians={model.num_gaussians}")
        tora.metric(name="train_loss", value=float(loss_value.item()), step_or_epoch=epoch)
        tora.metric(name="psnr", value=float(current_psnr), step_or_epoch=epoch)
        tora.metric(name="num_gaussians", value=float(model.num_gaussians), step_or_epoch=epoch)

        if model.num_gaussians < total_budget:
            error_map_mx = mx.sum(mx.abs(rendered.astype(mx.float32) - target_image.astype(mx.float32)), axis=-1)
            add_probabilities = build_probability_mass_from_error_map(error_map_mx)

            last_target_epoch = epochs - add_every
            if last_target_epoch < 1:
                last_target_epoch = 1

            epoch_index_one_based = (epoch + 1)
            is_add_epoch = (epoch_index_one_based % add_every == 0) or (epoch_index_one_based == last_target_epoch)
            if epoch_index_one_based > last_target_epoch:
                is_add_epoch = False

            if is_add_epoch:
                target_count = int(num_init + math.floor((min(epoch_index_one_based, last_target_epoch) * (total_budget - num_init)) / max(1, last_target_epoch)))
                target_count = min(target_count, total_budget)
                num_to_add = max(0, target_count - model.num_gaussians)
            else:
                num_to_add = 0

            if num_to_add > 0:
                add_indices = random_generator.choice(height * width, size=num_to_add, replace=True, p=add_probabilities)
                add_rows = (add_indices // width).astype(np.int32)
                add_cols = (add_indices % width).astype(np.int32)
                new_means_mx = mx.stack([mx.array(add_cols, dtype=mx.float32), mx.array(add_rows, dtype=mx.float32)], axis=-1)
                target_image_np = np.array(target_image)
                new_colors_np = target_image_np[add_rows, add_cols, :]
                new_colors_mx = mx.array(new_colors_np.astype(np.float32))
                model.add_gaussians(new_means_mx, new_colors_mx)
                loss_and_gradient_fn = nn.value_and_grad(model, sampled_l1_loss)
                optimizer = optim.Adam(learning_rate=lr)
                print(f"[densify] epoch={epoch+1} added={num_to_add}  total={model.num_gaussians}/{total_budget}")
                tora.metric(name="num_gaussians", value=float(model.num_gaussians), step_or_epoch=epoch)

        if save is not None:
            output_image = (mx.clip(rendered, 0.0, 1.0) * 255.0).astype(mx.uint8)
            output_image_np = np.array(output_image)
            cv2.imwrite(f"output/epoch_{epoch}_{save}", output_image_np[:, :, ::-1])

    tora.shutdown()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num_init", type=int, default=2000)
    parser.add_argument("--tile", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=10_000)
    parser.add_argument("--sample_mix", type=float, default=0.5,
                        help="Mixture weight for gradient-guided vs uniform sampling (0..1)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workspace_id", type=str, default=None)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--budget", type=int, default=None,
                        help="Total Gaussian budget. Defaults to num_init (no densification unless > num_init).")
    parser.add_argument("--add_every", type=int, default=500,
                        help="Add new Gaussians every N epochs and reach --budget by epoch (epochs - add_every). Uses Eq.(7) error magnitude sampling.")

    args = parser.parse_args()
    if not args.image.exists():
        raise FileNotFoundError("image not found!")

    main(
        image_path=args.image.as_posix(),
        epochs=args.epochs,
        lr=args.lr,
        num_init=args.num_init,
        tile=args.tile,
        num_samples=args.num_samples,
        sample_mix=args.sample_mix,
        seed=args.seed,
        workspace_id=args.workspace_id,
        save=args.save,
        budget=args.budget,
        add_every=args.add_every,
    )
