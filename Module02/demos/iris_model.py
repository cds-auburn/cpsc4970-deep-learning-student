"""Shared model definition and saved preprocessing for the Iris classifier."""
import json
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models" / "iris"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
LABELS = ["setosa", "versicolor", "virginica"]


def architecture():
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))


def prepare(measurements, metadata):
    """Standardize raw measurements in centimeters, shape (N, 4), with the saved scales."""
    x = torch.as_tensor(measurements, dtype=torch.float32)
    if x.ndim != 2 or x.shape[1] != 4 or not torch.isfinite(x).all():
        raise ValueError("Expected finite measurements with shape (N, 4)")
    mean = torch.tensor(metadata["mean"])
    std = torch.tensor(metadata["std"])
    return (x - mean) / std


def load_model():
    metadata = json.loads((BUNDLE / "metadata.json").read_text())
    model = architecture()
    model.load_state_dict(torch.load(BUNDLE / "weights.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()
    return model, metadata
