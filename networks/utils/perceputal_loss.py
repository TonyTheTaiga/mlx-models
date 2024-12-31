import mlx.nn as nn
import mlx.core as mx


class PerceptualLoss:
    def __init__(self, feature_extractor: nn.Module):
        self.feature_extractor = feature_extractor
        self.feature_extractor.train(False)
        mx.eval(self.feature_extractor.parameters())

    def __call__(self, x: mx.array, y: mx.array, layers: list[int] | None = None):
        # stack x and y and unstack after features are extracted?
        sample_features = self.feature_extractor.extract_features(x)  # pyright: ignore
        target_features = self.feature_extractor.extract_features(y)  # pyright: ignore

        if layers is None:
            layers = list(range(len(sample_features)))

        loss = 0
        for idx in layers:
            x_f, y_f = sample_features[idx], target_features[idx]
            loss += nn.losses.mse_loss(x_f, y_f, reduction="mean")

        return loss
