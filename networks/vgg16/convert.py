"""
Converts the pretrained torchvision weights into MLX format
"""

import numpy as np
import torch
import torchvision

FEATURE_DIM = 512


def convert():
    torch_weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1.get_state_dict()
    state_dict = {}
    for k, v in torch_weights.items():
        if len(v.shape) == 4:
            v = v.permute(0, 3, 2, 1)
        elif len(v.shape) == 2 and "classifier.0" in k:
            in_features = v.shape[1]
            spatial_dim = int(np.sqrt((in_features / FEATURE_DIM)))
            v = v.reshape(v.shape[0], FEATURE_DIM, spatial_dim, spatial_dim)
            v = torch.permute(v, (0, 2, 3, 1))
            v = v.reshape(v.shape[0], -1)

        k = k.split(".")
        k.insert(1, "layers")
        k = ".".join(k)
        state_dict[k] = v.numpy()

    np.savez("weights", **state_dict)


if __name__ == "__main__":
    convert()
