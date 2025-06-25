from pathlib import Path
from uuid import uuid4

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import cv2
import numpy as np
from mlx.utils import tree_flatten

from networks.ddpm.model import UNET
from tora import Tora  # pyright: ignore


WORKSPACE_ID = "5f0ae752-9d6d-4c67-b0ba-2fd601a83831"
MNIST_PATH = Path("/Users/taigaishida/workspace/mlx-models/mnist/")
T = 500
DESCRIPTION = "t_dim = 64"
BETA_MIN = 1e-4
BETA_MAX = 2e-2
BETA = mx.linspace(BETA_MIN, BETA_MAX, T)
ALPHA = 1 - BETA
ALPHABAR = mx.cumprod(ALPHA, axis=0)
ALPHABAR_SQRT = mx.sqrt(ALPHABAR)
ALPHABAR_SQRT_OM = mx.sqrt(1 - ALPHABAR)

ALPHABAR_PREV = mx.roll(ALPHABAR, 1)
ALPHABAR_PREV[0] = 1.0
POST_VAR = BETA * (1 - ALPHABAR_PREV) / (1 - ALPHABAR)
POST_VAR[0] = 0.0
C1 = (mx.sqrt(ALPHABAR_PREV) * BETA) / (1 - ALPHABAR)
C2 = (mx.sqrt(ALPHA) * (1 - ALPHABAR_PREV)) / (1 - ALPHABAR)


def load_a_image(path: str | Path, read_flag=cv2.IMREAD_COLOR_BGR):
    np_img = cv2.imread(path, read_flag)  # pyright: ignore
    np_img = np_img / 127.5 - 1.0
    if len(np_img.shape) == 2:
        np_img = np.expand_dims(np_img, 2)

    return np_img


def load_mnist() -> dict[str, mx.array]:
    train = []
    train_labels = []
    for p in (MNIST_PATH / "training").rglob("**/*.png"):
        train_labels.append(int(p.parent.name))
        train.append(load_a_image(p, cv2.IMREAD_GRAYSCALE))

    train_labels_arr = np.array(train_labels)
    train_arr = np.array(train)

    val = []
    val_labels = []
    for p in (MNIST_PATH / "testing").rglob("**/*.png"):
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
):
    culm_loss = 0.0
    nsamples = 0
    for x_clean in dataloader(dataset, batch_size=64):
        t = mx.random.randint(0, T, (x_clean.shape[0],), dtype=mx.int32)
        noisy, eps = add_noise(x_clean, t)
        loss = mx.mean((model(noisy, t) - eps) ** 2).item()
        culm_loss += loss * x_clean.shape[0]  # pyright: ignore
        nsamples += x_clean.shape[0]

    return culm_loss / nsamples


def dataloader(data, batch_size):
    idx = mx.random.permutation(len(data))
    for start in range(0, len(data), batch_size):
        yield data[idx[start : start + batch_size]]


def add_noise(x: mx.array, t: mx.array):
    eps = mx.random.normal(shape=x.shape, dtype=mx.float32)
    sqrt_ab = ALPHABAR_SQRT[t][:, None, None, None]
    sqrt_one = ALPHABAR_SQRT_OM[t][:, None, None, None]
    x_t = sqrt_ab * x + sqrt_one * eps
    return x_t, eps


def sample_image(model: UNET, show_progress=False):
    x = mx.random.normal(shape=(1, 28, 28, 1), dtype=mx.float32)

    if show_progress:
        print("Starting denoising process...")
        display((x + 1) / 2)
        print(f"Step: {T} (pure noise)")

    for _t in reversed(range(T)):
        t = mx.full(shape=(1,), vals=_t, dtype=mx.int32)
        noise = model(x, t)
        clean = (x - ALPHABAR_SQRT_OM[t][:, None, None, None] * noise) / ALPHABAR_SQRT[t]
        mean = C1[t][:, None, None, None] * clean + C2[t][:, None, None, None] * x

        if _t > 0:
            posterior_variance_t = POST_VAR[t][:, None, None, None]
            _noise_sample = mx.random.normal(shape=x.shape, dtype=mx.float32)
            x = mean + mx.sqrt(posterior_variance_t) * _noise_sample
        else:
            x = mean

        if show_progress and (_t % 100 == 0 or _t < 10):
            print(f"\nStep: {_t}")
            display((x + 1) / 2)

    return x


def main():
    batch_size = 8
    epochs = 100
    learning_rate = 1e-4
    t_dim = 64
    input_dim = 1
    unet = UNET(input_dim, T, t_dim)
    mx.eval(unet.parameters())
    num_params = sum(v.size for _, v in tree_flatten(unet.parameters()))
    tora = Tora.create_experiment(
        name=f"DDPM_MNIST_{uuid4().hex[:3]}",
        description=DESCRIPTION,
        hyperparams={
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "t": T,
            "t_dim": t_dim,
            "beta_min": BETA_MIN,
            "beta_max": BETA_MAX,
            "num_params": num_params,
        },
    )
    tora.max_buffer_len = 1

    optimizer = optim.AdamW(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(unet, loss_fn)
    dataset = load_mnist()
    steps = 0
    for epoch in range(epochs):
        culm_loss = 0
        num_samples = 0

        for step, x_clean in enumerate(dataloader(dataset["train"], batch_size)):
            steps += step
            t = mx.random.randint(0, T, (x_clean.shape[0],), dtype=mx.int32)
            x_noisy, eps = add_noise(x_clean, t)
            loss, grads = loss_and_grad_fn(unet, x_noisy, eps, t)
            optimizer.update(unet, grads)
            mx.eval(unet.parameters(), optimizer.state)
            culm_loss += loss.item() * (x_clean.shape[0])
            num_samples += x_clean.shape[0]

        epoch_loss = culm_loss / num_samples
        tora.log(name="epoch_loss", value=float(epoch_loss), step=epoch)

        epoch_eval_loss = eval_fn(unet, dataset["val"])
        tora.log(name="epoch_eval_loss", value=float(epoch_eval_loss), step=epoch)

        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"\nGenerating sample with step-by-step visualization (epoch {epoch + 1}):")
            sample_image(unet, show_progress=True)
        else:
            samples = [sample_image(unet) for _ in range(3)]
            samples_mx = mx.concat(samples, axis=1)
            display(samples_mx)


if __name__ == "__main__":
    main()
