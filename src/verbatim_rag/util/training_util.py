import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


def compute_class_weights(labels: list[int]) -> torch.Tensor:
    classes = np.array([0, 1], dtype=int)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)
