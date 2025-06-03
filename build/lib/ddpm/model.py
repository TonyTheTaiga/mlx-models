import mlx.core as mx
import mlx.nn as nn


class UNET(nn.Module):
    def __init__(self):
        self.stem = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=2)

    def __call__(self, x):
        x = self.stem(x)
        return x


if __name__ == "__main__":
    network = UNET()
    out = network(mx.zeros((1, 28, 28, 1)))
    print(out.shape)
