import mlx.core as mx
import mlx.nn as nn


class SSDHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes
        self.loc_heads = [
            nn.Conv2d(512, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(1024, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * 4, kernel_size=3, padding=1),
        ]
        self.conf_heads = [
            nn.Conv2d(512, 4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(1024, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(512, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 6 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),
            nn.Conv2d(256, 4 * num_classes, kernel_size=3, padding=1),
        ]

    def __call__(self, features):
        locs, confs = [], []
        for feature, lhead, chead in zip(features, self.loc_heads, self.conf_heads):
            locs.append(lhead(feature).reshape((feature.shape[0], -1, 4)))
            confs.append(chead(feature).reshape((feature.shape[0], -1, self.num_classes)))

        locs = mx.concat(locs, axis=1)
        confs = mx.concat(confs, axis=1)

        return locs, confs


class SSD300(nn.Module):
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
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            # extras starting here
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(1024, 256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
        )

        self.norm = nn.RMSNorm(512)
        self.head = SSDHead(num_classes)

    def __call__(self, x: mx.array):
        features = []
        #   22 → conv4_3 (37×37)
        #   35 → conv7   (19×19)
        #   39 → conv8_2 (10×10)
        #   43 → conv9_2 (5×5)
        #   47 → conv10_2 (3×3)
        #   51 → conv11_2 (1×1)
        capture_idxs = {22, 35, 39, 43, 47, 51}

        for idx, layer in enumerate(self.features.layers):
            x = layer(x)
            if idx in capture_idxs:
                if idx == 22:
                    features.append(self.norm(x))
                else:
                    features.append(x)

        return self.head(features)
