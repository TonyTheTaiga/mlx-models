import mlx.core as mx
import mlx.nn as nn
from numpy import isin


def adaptive_avg_pool_2d(x: mx.array, output_size: tuple[int, int]):
    B, H, W, C = x.shape
    (o_H, o_W) = output_size
    x = x.reshape(B, o_H, H // o_H, o_W, W // o_W, C)
    x = mx.mean(x, axis=(2, 4))
    return x


class VGG16(nn.Module):
    def __init__(self, num_classes, dropout=0.1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

    def __call__(self, x: mx.array):
        x = self.features(x)
        x = adaptive_avg_pool_2d(x, (7, 7))
        x = x.reshape(x.shape[0], -1)
        return self.classifier(x)

    def extract_features(self, x: mx.array):
        features = []
        for layer in self.features.children()["layers"]:
            if isinstance(layer, nn.MaxPool2d):
                features.append(x)

            x = layer(x)

        return features


if __name__ == "__main__":
    import mlx.core as mx
    import numpy as np
    import cv2

    def load_image(path: str):
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Failed to load image at {path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
        image = image / 255.0

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        image = (image - mean) / std
        return image

    model = VGG16(1000)
    model.load_weights("weights.npz")
    image = load_image("test.jpeg")
    image = mx.array(image)
    image = mx.expand_dims(image, 0)
    model.extract_features(image)
    # print(mx.argmax(mx.softmax(model(image), -1), -1))

    # x3 = np.array([[[[1.0], [2.0], [3.0], [4.0]], [[5.0], [6.0], [7.0], [8.0]]]])  # Shape: (1, 2, 4, 1)
    # x3 = mx.array(x3)
    # result3 = adaptive_avg_pool_2d(x3, (2, 2))
    # expected3 = np.array([[[[1.5], [3.5]], [[5.5], [7.5]]]])
    # expected3 = mx.array(expected3)

    # print(result3[0])
    # print(expected3[0])
