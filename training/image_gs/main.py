import cv2
import mlx.core as mx

from networks.image_gs import Gaussian2D, get_tiles, image_gradient_map


def main(image_path: str):
    image = cv2.imread(image_path, flags=cv2.IMREAD_COLOR_RGB)
    if image is None:
        raise RuntimeError("failed to load image")

    image_array = mx.array(image)
    tiles, coords = get_tiles(image_array, 16)
    gradient_map = image_gradient_map(image_array)


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
