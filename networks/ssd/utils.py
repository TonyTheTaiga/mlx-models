import mlx.core as mx
import math


def linterpolate(min_scale, max_scale, m, steps):
    Δ = (max_scale - min_scale) / (m - 1)
    return [min_scale + Δ * (k - 1) for k in range(1, steps + 1)]


def generate_anchors(ratios: list[float], feature_map_sizes: list[int]):
    scales = linterpolate(0.2, 0.9, len(feature_map_sizes), len(feature_map_sizes) + 1)
    priors_all = []
    for feature_map_index, K in enumerate(feature_map_sizes):
        whs = []
        scale = scales[feature_map_index]
        for ratio in ratios:
            if (ratio == 3 or ratio == 1 / 3) and feature_map_index in [0, 4, 5]:
                continue

            whs.append((scale * math.sqrt(ratio), scale / math.sqrt(ratio)))

        whs.append(
            (
                math.sqrt(scale * scales[feature_map_index + 1]),
                math.sqrt(scale * scales[feature_map_index + 1]),
            )
        )

        for i in range(K):
            for j in range(K):
                pos_i, pos_j = (i + 0.5) / K, (j + 0.5) / K
                priors_all.extend([(pos_i, pos_j, w, h) for (w, h) in whs])

    anchors = mx.array(priors_all)
    return anchors
