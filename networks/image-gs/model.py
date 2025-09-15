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

        self.mean = mx.array([0, 0])
        self.theta = mx.random.uniform(low=0, high=math.pi, shape=())
        self.log_scales = mx.random.uniform(low=-5.0, high=-2.0, shape=(2,))
        self.c = mx.random.uniform(low=0.0, high=1.0, shape=(channels,))

    def __call__(self, x):
        rm = rotation_matrix_2d(mx.clip(self.theta, 0.0, math.pi))
        sm = scaling_matrix(mx.exp(self.log_scales))
        cm = cov_from_r_and_s(rm, sm) + 1e-6 * mx.eye(2)
        d = x - self.mean
        return mx.exp(-0.5 * (d @ mx.linalg.solve(cm, d, stream=mx.cpu))) * self.c  # pright: ignore


if __name__ == "__main__":
    g = Gaussian2D(3)
    X = mx.array([420, 69])

    print(g(X))
