import math

import mlx.core as mx
import mlx.nn as nn


def rotation_matrix_2d(theta):
    cos = mx.cos(theta)
    sin = mx.sin(theta)
    return mx.stack([mx.stack([cos, -sin]), mx.stack([sin, cos])])


def scaling_matrix(v: mx.array):
    return mx.diag(v)


def cov_from_r_and_s(r: mx.array, s: mx.array):
    return r @ s @ mx.transpose(s) @ mx.transpose(r)


class Gaussian2D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()

        self.mean = mx.array([0.0, 0.0], dtype=mx.float16)
        self.theta = mx.array(0.0, dtype=mx.float16)
        self.raw_s = mx.random.uniform(low=-3.0, high=3.0, shape=(2,), dtype=mx.float16)
        self.c = mx.random.uniform(low=0.0, high=1.0, shape=(channels,), dtype=mx.float16)

    def __call__(self, x):
        rm = rotation_matrix_2d(math.pi * mx.sigmoid(self.theta))
        eps = 1e-6
        inv_s = 1.0 / (nn.softplus(self.raw_s) + eps)
        sm = scaling_matrix(inv_s)
        P = cov_from_r_and_s(rm, sm) + eps * mx.eye(2)
        d = x - self.mean
        q = mx.sum((d @ P) * d, axis=-1)
        g = mx.exp(-0.5 * q)
        return g * self.c


if __name__ == "__main__":
    g = Gaussian2D(3)
    X = mx.array([420, 69], dtype=mx.float16)
    print(g(X))
