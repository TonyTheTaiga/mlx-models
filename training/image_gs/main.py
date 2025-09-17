import cv2
import mlx.core as mx

from networks.image_gs import build_model


def main(image_path: str):
    image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR_RGB)
    if image is None:
        raise RuntimeError("failed to load image")

    image_array = mx.array(image)
    model = build_model(image_array, samples=3)
    model(mx.random.randint(low=0, high=255, shape=(2048, 2048, 3)))


def loss_fn(model, tiles): ...


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    image_path: Path = parser.parse_args().image
    if not image_path.exists():
        raise FileNotFoundError("image not found!")

    main(image_path.as_posix())
