import math
from pathlib import Path
from uuid import uuid4

import cv2
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from loss import l1_ssim_loss
from mlx.utils import tree_flatten
from tora import Tora

from networks.image_gs import build_model


def srgb_to_linear(x: mx.array) -> mx.array:
    x32 = x.astype(mx.float32)
    a = mx.array(0.04045, dtype=mx.float32)
    return mx.where(
        x32 <= a, x32 / mx.array(12.92, dtype=mx.float32), ((x32 + 0.055) / 1.055) ** 2.4
    )


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


def compute_psnr(x: mx.array, y: mx.array, max_val: float = 1.0) -> float:
    x32 = x.astype(mx.float32)
    y32 = y.astype(mx.float32)
    diff2 = (x32 - y32) ** 2
    mask = mx.isfinite(diff2)
    valid = mx.sum(mask).item()
    if valid == 0:
        return 0.0
    diff2 = mx.where(mask, diff2, mx.array(0.0, dtype=mx.float32))
    mse = (mx.sum(diff2).item()) / float(valid)
    if not math.isfinite(mse) or mse <= 1e-12:
        return 99.0
    return float(20.0 * math.log10(max_val) - 10.0 * math.log10(mse))


def main(
    image_path: str,
    epochs: int = 2,
    tile: int = 16,
    sample_mix: float = 0.3,
    seed: int | None = 42,
    workspace_id: str | None = None,
    save: Path | None = None,
    budget: int = 0,
    linear_error: bool = False,
    no_duplicate_additions: bool = False,
    top_k: int = 10,
):
    image_bgr = cv2.imread(image_path, flags=cv2.IMREAD_COLOR_RGB)
    if image_bgr is None:
        raise RuntimeError("failed to load image")

    target_image = mx.array(image_bgr).astype(mx.float32) / 255.0
    total_budget = int(budget)
    initial_gaussians = max(1, total_budget // 2)

    model = build_model(
        target_image, num_initial_gaussians=initial_gaussians, tile_size=tile, top_k=top_k
    )
    model.set_dtype(mx.float16)
    mx.eval(model.parameters())
    model(target_image)

    random_generator = np.random.default_rng(seed)
    loss_and_gradient_fn = nn.value_and_grad(model, l1_ssim_loss)

    lr_mu = 5e-4
    lr_c = 5e-3
    lr_s = 2e-3
    lr_theta = 2e-3
    adam_eps = 1e-4
    opt_mu = optim.Adam(learning_rate=lr_mu, eps=adam_eps)
    opt_c = optim.Adam(learning_rate=lr_c, eps=adam_eps)
    opt_s = optim.Adam(learning_rate=lr_s, eps=adam_eps)
    opt_theta = optim.Adam(learning_rate=lr_theta, eps=adam_eps)
    total_budget = max(total_budget, int(2 * model.num_gaussians))

    ramp_end = max(1, int(0.4 * epochs))
    exp_name = f"ImageGS_{uuid4().hex[:3]}"

    tora = Tora.create_experiment(
        name=exp_name,
        workspace_id=workspace_id,
        description="Image Gaussian Splatting: single-image reconstruction",
        hyperparams={
            "epochs": epochs,
            "lr_mu": 5e-4,
            "lr_c": 5e-3,
            "lr_s": 2e-3,
            "lr_theta": 2e-3,
            "budget": total_budget,
            "initial_gaussians": initial_gaussians,
            "tile_size": tile,
            "sample_mix": sample_mix,
            "ramp_end_epochs": ramp_end,
        },
        max_buffer_len=1,
    )

    height, width, _ = image_bgr.shape
    for epoch in range(epochs):
        tora.metric(name="num_gaussians", value=float(model.num_gaussians), step_or_epoch=epoch)

        try:
            mean32 = model.mean.astype(mx.float32)
            nan_mask = ~mx.isfinite(mean32)
            if mx.sum(nan_mask).item() > 0:
                safe_x = mx.where(nan_mask[:, 0], mx.array(0.5, dtype=mx.float32), mean32[:, 0])
                safe_y = mx.where(nan_mask[:, 1], mx.array(0.5, dtype=mx.float32), mean32[:, 1])
                mean32 = mx.stack([safe_x, safe_y], axis=-1)
            clamped_x = mx.minimum(
                mx.maximum(mean32[:, 0], mx.array(0.0, dtype=mx.float32)),
                mx.array(1.0, dtype=mx.float32),
            )
            clamped_y = mx.minimum(
                mx.maximum(mean32[:, 1], mx.array(0.0, dtype=mx.float32)),
                mx.array(1.0, dtype=mx.float32),
            )
            model.mean = mx.stack([clamped_x, clamped_y], axis=-1).astype(model.mean.dtype)
            model.color = mx.where(
                mx.isfinite(model.color.astype(mx.float32)),
                model.color.astype(mx.float32),
                mx.array(0.5, dtype=mx.float32),
            ).astype(model.color.dtype)
            model.inv_scale = mx.where(
                mx.isfinite(model.inv_scale.astype(mx.float32)),
                mx.clip(model.inv_scale.astype(mx.float32), 1e-3, 1.0),
                mx.array(0.2, dtype=mx.float32),
            ).astype(model.inv_scale.dtype)
            model.theta = mx.where(
                mx.isfinite(model.theta.astype(mx.float32)),
                mx.clip(model.theta.astype(mx.float32), -8.0, 8.0),
                mx.array(0.0, dtype=mx.float32),
            ).astype(model.theta.dtype)
        except Exception:
            pass
        (loss_value, rendered), gradients = loss_and_gradient_fn(model, target_image)

        if not math.isfinite(loss_value.item()):
            raise RuntimeError("non-finite loss detected")
        else:
            grads_dict = dict(tree_flatten(gradients))

            def pick(pred):
                return {k: v for k, v in grads_dict.items() if pred(k)}

            for k, g in grads_dict.items():
                if mx.sum(~mx.isfinite(g.astype(mx.float32))).item() > 0:
                    raise RuntimeError(f"non-finite gradients detected in {k}")

            def clip_grads(d, bound=10.0):
                out = {}
                b = mx.array(bound, dtype=mx.float32)
                for k, g in d.items():
                    g32 = g.astype(mx.float32)
                    g32 = mx.minimum(mx.maximum(g32, -b), b)
                    out[k] = g32
                return out

            grads_mu = clip_grads(pick(lambda n: "mean" in n))
            grads_c = clip_grads(pick(lambda n: "color" in n))
            grads_s = clip_grads(pick(lambda n: "inv_scale" in n))
            grads_theta = clip_grads(pick(lambda n: "theta" in n))

            opt_mu.update(model, grads_mu)
            opt_c.update(model, grads_c)
            opt_s.update(model, grads_s)
            opt_theta.update(model, grads_theta)

            mean32 = model.mean.astype(mx.float32)
            clamped_x = mx.minimum(
                mx.maximum(mean32[:, 0], mx.array(0.0, dtype=mx.float32)),
                mx.array(1.0, dtype=mx.float32),
            )
            clamped_y = mx.minimum(
                mx.maximum(mean32[:, 1], mx.array(0.0, dtype=mx.float32)),
                mx.array(1.0, dtype=mx.float32),
            )
            model.mean = mx.stack([clamped_x, clamped_y], axis=-1).astype(model.mean.dtype)
            model.color = mx.clip(model.color.astype(mx.float32), 0.0, 1.0).astype(
                model.color.dtype
            )
            model.inv_scale = mx.clip(model.inv_scale.astype(mx.float32), 1e-3, 1.0).astype(
                model.inv_scale.dtype
            )
            model.theta = mx.clip(model.theta.astype(mx.float32), -8.0, 8.0).astype(
                model.theta.dtype
            )

        mx.eval(model.parameters(), opt_mu.state, opt_c.state, opt_s.state, opt_theta.state)

        current_psnr = compute_psnr(rendered, target_image)
        print(
            f"epoch {epoch + 1}/{epochs}  loss={loss_value.item():.6f}  psnr={current_psnr:.2f} dB  gaussians={model.num_gaussians}"
        )
        tora.metric(name="train_loss", value=float(loss_value.item()), step_or_epoch=epoch)
        tora.metric(name="psnr", value=float(current_psnr), step_or_epoch=epoch)

        if model.num_gaussians < total_budget:
            if linear_error:
                x_lin = srgb_to_linear(rendered)
                y_lin = srgb_to_linear(target_image)
                error_map_mx = mx.sum(mx.abs(x_lin - y_lin), axis=-1)
            else:
                error_map_mx = mx.sum(
                    mx.abs(rendered.astype(mx.float32) - target_image.astype(mx.float32)), axis=-1
                )
            add_probabilities = build_probability_mass_from_error_map(error_map_mx)

            increments_done = min(((epoch + 1) // 500), 4)
            start = total_budget // 2
            per_increment = max(1, total_budget // 8)
            target_count = int(min(total_budget, start + increments_done * per_increment))
            num_to_add = max(0, target_count - model.num_gaussians)

            if num_to_add > 0:
                population = height * width
                if no_duplicate_additions and num_to_add <= population:
                    add_indices = random_generator.choice(
                        population, size=num_to_add, replace=False, p=add_probabilities
                    )
                else:
                    add_indices = random_generator.choice(
                        population, size=num_to_add, replace=True, p=add_probabilities
                    )

                add_rows = (add_indices // width).astype(np.int32)
                add_cols = (add_indices % width).astype(np.int32)
                new_means_mx = mx.stack(
                    [
                        mx.array(add_cols, dtype=mx.float32) / mx.array(width, dtype=mx.float32),
                        mx.array(add_rows, dtype=mx.float32) / mx.array(height, dtype=mx.float32),
                    ],
                    axis=-1,
                )
                target_image_np = np.array(target_image)
                new_colors_np = target_image_np[add_rows, add_cols, :]
                new_colors_mx = mx.array(new_colors_np.astype(np.float32))
                model.add_gaussians(new_means_mx, new_colors_mx)
                mx.eval(model.parameters())

                loss_and_gradient_fn = nn.value_and_grad(model, l1_ssim_loss)
                lr_mu = 5e-4
                lr_c = 5e-3
                lr_s = 2e-3
                lr_theta = 2e-3
                opt_mu = optim.Adam(learning_rate=lr_mu, eps=adam_eps)
                opt_c = optim.Adam(learning_rate=lr_c, eps=adam_eps)
                opt_s = optim.Adam(learning_rate=lr_s, eps=adam_eps)
                opt_theta = optim.Adam(learning_rate=lr_theta, eps=adam_eps)

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
    parser.add_argument("--tile", type=int, default=16)
    parser.add_argument(
        "--sample_mix",
        type=float,
        default=0.3,
        help="Mixture weight for gradient-guided vs uniform sampling (0..1)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workspace_id", type=str, default=None)
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument(
        "--linear_error",
        action="store_true",
        help="Compute densification error map in linear RGB instead of sRGB",
    )
    parser.add_argument(
        "--no_duplicate_additions",
        action="store_true",
        help="Sample new Gaussians without replacement within each densification step",
    )
    parser.add_argument(
        "--budget",
        type=int,
        required=True,
        help="Total Gaussian budget Ng. Initial count = Ng/2.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        required=False,
        default=10,
        help="Top K gaussians to use for each tile position",
    )

    args = parser.parse_args()
    if not args.image.exists():
        raise FileNotFoundError("image not found!")

    main(
        image_path=args.image.as_posix(),
        epochs=args.epochs,
        tile=args.tile,
        sample_mix=args.sample_mix,
        seed=args.seed,
        workspace_id=args.workspace_id,
        save=args.save,
        budget=args.budget,
        linear_error=args.linear_error,
        no_duplicate_additions=args.no_duplicate_additions,
        top_k=args.top_k,
    )
