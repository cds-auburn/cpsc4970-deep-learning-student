"""Shared model definition and frozen preprocessing contract."""
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "models" / "heart"
CONTINUOUS = ["age", "trestbps", "chol", "thalach"]
LABELS = ["absence", "presence"]


def architecture():
    return nn.Sequential(nn.Linear(8, 12), nn.ReLU(), nn.Linear(12, 2))


def prepare(records, metadata):
    """Accept named CSV records; never use the target as an input."""
    numeric = torch.tensor([[float(row[k]) for k in metadata["continuous"]]
                            for row in records], dtype=torch.float32)
    category = torch.tensor([int(row["cp"]) for row in records])
    if not torch.isfinite(numeric).all() or not ((category >= 1) & (category <= 4)).all():
        raise ValueError("Expected finite measurements and cp codes 1 through 4")
    mean = torch.tensor(metadata["mean"])
    std = torch.tensor(metadata["std"])
    numeric = (numeric - mean) / std
    pain = F.one_hot(category - 1, num_classes=4).to(torch.float32)
    return torch.cat([numeric, pain], dim=1)


def load_model():
    metadata = json.loads((BUNDLE / "metadata.json").read_text())
    model = architecture()
    model.load_state_dict(torch.load(BUNDLE / "weights.pt", map_location="cpu",
                                     weights_only=True))
    model.eval()
    return model, metadata
