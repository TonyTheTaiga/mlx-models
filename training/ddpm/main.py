from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import cv2
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import yaml
from mlx.utils import tree_flatten
from tora import Tora  # pyright: ignore

from networks.ddpm.model import UNET

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
    mnist_path = resolve_path(cfg.get("dataset_root"), base) or Path(
        "/Users/taigaishida/workspace/mlx-models/mnist/"
    )
    return SimpleNamespace(
        mnist_path=mnist_path,
        workspace_id=cfg.get("workspace_id", "5f0ae752-9d6d-4c67-b0ba-2fd601a83831"),
        description=cfg.get("description", "t_dim = 64"),
        epochs=int(cfg.get("epochs", 100)),
        batch_size=int(cfg.get("batch_size", 8)),
        learning_rate=float(cfg.get("learning_rate", 1e-4)),
        t_steps=int(cfg.get("t_steps", 500)),
        t_dim=int(cfg.get("t_dim", 64)),
        input_channels=int(cfg.get("input_channels", 1)),
        beta_min=float(cfg.get("beta_min", 1e-4)),
        beta_max=float(cfg.get("beta_max", 2e-2)),
    )


def build_diffusion_tables(cfg: SimpleNamespace) -> SimpleNamespace:
    beta = mx.linspace(cfg.beta_min, cfg.beta_max, cfg.t_steps)
    alpha = 1 - beta
    alphabar = mx.cumprod(alpha, axis=0)
    alphabar_sqrt = mx.sqrt(alphabar)
    alphabar_sqrt_om = mx.sqrt(1 - alphabar)

    alphabar_prev = mx.roll(alphabar, 1)
    alphabar_prev[0] = 1.0
    post_var = beta * (1 - alphabar_prev) / (1 - alphabar)
    post_var[0] = 0.0
    c1 = (mx.sqrt(alphabar_prev) * beta) / (1 - alphabar)
    c2 = (mx.sqrt(alpha) * (1 - alphabar_prev)) / (1 - alphabar)

    return SimpleNamespace(
        beta=beta,
        alpha=alpha,
        alphabar=alphabar,
        alphabar_sqrt=alphabar_sqrt,
        alphabar_sqrt_om=alphabar_sqrt_om,
        alphabar_prev=alphabar_prev,
        post_var=post_var,
        c1=c1,
        c2=c2,
    )


def load_a_image(path: str | Path, read_flag=cv2.IMREAD_COLOR_BGR):
    np_img = cv2.imread(path, read_flag)  # pyright: ignore
    np_img = np_img / 127.5 - 1.0
    if len(np_img.shape) == 2:
        np_img = np.expand_dims(np_img, 2)

    return np_img


def load_mnist(cfg: SimpleNamespace) -> dict[str, mx.array]:
    train = []
    train_labels = []
    for p in (cfg.mnist_path / "training").rglob("**/*.png"):
        train_labels.append(int(p.parent.name))
        train.append(load_a_image(p, cv2.IMREAD_GRAYSCALE))

    train_labels_arr = np.array(train_labels)
    train_arr = np.array(train)

    val = []
    val_labels = []
    for p in (cfg.mnist_path / "testing").rglob("**/*.png"):
        val_labels.append(int(p.parent.name))
        val.append(load_a_image(p, cv2.IMREAD_GRAYSCALE))

    val_labels_arr = np.array(val_labels)
    val_arr = np.array(val)

    return {
        "train": mx.array(train_arr),
        "train_labels": mx.array(train_labels_arr),
        "val": mx.array(val_arr),
        "val_labels": mx.array(val_labels_arr),
    }


def display(img: mx.array | np.ndarray) -> None:
    """
    Renders a grayscale image to the terminal using ANSI 24-bit true color codes.
    Assumes img values are in [0, 1].
    """
    RESET = "\x1b[0m"

    if isinstance(img, mx.array):
        img = np.array(img)

    img = img.squeeze()

    if img.dtype != np.float32 and img.dtype != np.float64:
        img = img.astype(np.float32)
    img = np.clip(img, 0.0, 1.0)  # Ensure values are within [0, 1]

    for row in img:
        line = ""
        for px_val in row:
            # Scale pixel value to 0-255 for RGB components
            g = int(px_val * 255)
            # Use foreground color for the block character
            line += f"\x1b[38;2;{g};{g};{g}m█{RESET}"
            # You could also use background color for 2x vertical resolution
            # line += f"\x1b[48;2;{g};{g};{g}m \x1b[0m" # This uses a space and background color
        print(line)


def loss_fn(model: UNET, noisey_image: mx.array, eps: mx.array, t: mx.array):
    return mx.mean((model(noisey_image, t) - eps) ** 2)


def eval_fn(
    model: UNET,
    dataset,
    cfg: SimpleNamespace,
    diff: SimpleNamespace,
):
    culm_loss = 0.0
    nsamples = 0
    for x_clean in dataloader(dataset, batch_size=cfg.batch_size):
        t = mx.random.randint(0, cfg.t_steps, (x_clean.shape[0],), dtype=mx.int32)
        noisy, eps = add_noise(x_clean, t, diff)
        loss = mx.mean((model(noisy, t) - eps) ** 2).item()
        culm_loss += loss * x_clean.shape[0]  # pyright: ignore
        nsamples += x_clean.shape[0]

    return culm_loss / nsamples


def dataloader(data, batch_size):
    idx = mx.random.permutation(len(data))
    for start in range(0, len(data), batch_size):
        yield data[idx[start : start + batch_size]]


def add_noise(x: mx.array, t: mx.array, diff: SimpleNamespace):
    eps = mx.random.normal(shape=x.shape, dtype=mx.float32)
    sqrt_ab = diff.alphabar_sqrt[t][:, None, None, None]
    sqrt_one = diff.alphabar_sqrt_om[t][:, None, None, None]
    x_t = sqrt_ab * x + sqrt_one * eps
    return x_t, eps


def sample_image(
    model: UNET,
    cfg: SimpleNamespace,
    diff: SimpleNamespace,
    show_progress=False,
):
    x = mx.random.normal(shape=(1, 28, 28, cfg.input_channels), dtype=mx.float32)

    if show_progress:
        print("Starting denoising process...")
        display((x + 1) / 2)
        print(f"Step: {cfg.t_steps} (pure noise)")

    for _t in reversed(range(cfg.t_steps)):
        t = mx.full(shape=(1,), vals=_t, dtype=mx.int32)
        noise = model(x, t)
        clean = (x - diff.alphabar_sqrt_om[t][:, None, None, None] * noise) / diff.alphabar_sqrt[t]
        mean = diff.c1[t][:, None, None, None] * clean + diff.c2[t][:, None, None, None] * x

        if _t > 0:
            posterior_variance_t = diff.post_var[t][:, None, None, None]
            _noise_sample = mx.random.normal(shape=x.shape, dtype=mx.float32)
            x = mean + mx.sqrt(posterior_variance_t) * _noise_sample
        else:
            x = mean

        if show_progress and (_t % 100 == 0 or _t < 10):
            print(f"\nStep: {_t}")
            display((x + 1) / 2)

    return x


def main(cfg: SimpleNamespace | None = None):
    cfg = cfg or build_config()
    diff = build_diffusion_tables(cfg)

    unet = UNET(cfg.input_channels, cfg.t_steps, cfg.t_dim)
    mx.eval(unet.parameters())
    num_params = sum(v.size for _, v in tree_flatten(unet.parameters()))
    tora = Tora.create_experiment(
        name=f"DDPM_MNIST_{uuid4().hex[:3]}",
        description=cfg.description,
        hyperparams={
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "learning_rate": cfg.learning_rate,
            "t": cfg.t_steps,
            "t_dim": cfg.t_dim,
            "beta_min": cfg.beta_min,
            "beta_max": cfg.beta_max,
            "num_params": num_params,
        },
        workspace_id=cfg.workspace_id,
    )
    tora.max_buffer_len = 1

    optimizer = optim.AdamW(learning_rate=cfg.learning_rate)
    loss_and_grad_fn = nn.value_and_grad(unet, loss_fn)
    dataset = load_mnist(cfg)
    steps = 0
    for epoch in range(cfg.epochs):
        culm_loss = 0
        num_samples = 0

        for step, x_clean in enumerate(dataloader(dataset["train"], cfg.batch_size)):
            steps += step
            t = mx.random.randint(0, cfg.t_steps, (x_clean.shape[0],), dtype=mx.int32)
            x_noisy, eps = add_noise(x_clean, t, diff)
            loss, grads = loss_and_grad_fn(unet, x_noisy, eps, t)
            optimizer.update(unet, grads)
            mx.eval(unet.parameters(), optimizer.state)
            culm_loss += loss.item() * (x_clean.shape[0])
            num_samples += x_clean.shape[0]

        epoch_loss = culm_loss / num_samples
        tora.log(name="epoch_loss", value=float(epoch_loss), step=epoch)

        epoch_eval_loss = eval_fn(unet, dataset["val"], cfg, diff)
        tora.log(name="epoch_eval_loss", value=float(epoch_eval_loss), step=epoch)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"\nGenerating sample with step-by-step visualization (epoch {epoch + 1}):")
            sample_image(unet, cfg, diff, show_progress=True)
        else:
            samples = [sample_image(unet, cfg, diff) for _ in range(3)]
            samples_mx = mx.concat(samples, axis=1)
            display(samples_mx)


if __name__ == "__main__":
    main()
