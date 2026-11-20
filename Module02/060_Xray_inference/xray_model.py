"""Offline loading and deterministic preprocessing for the image demo."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchxrayvision as xrv

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"


def prepare_image(path):
    """The bundled files are 8-bit grayscale PNGs, not arbitrary DICOM images."""
    with Image.open(path) as source:
        if source.mode != "L":
            raise ValueError("Expected an 8-bit grayscale PNG (Pillow mode L)")
        pixels = np.array(source)
    image = xrv.datasets.normalize(pixels, 255)
    image = image[None, ...]  # (1, H, W): one grayscale channel.
    image = xrv.datasets.XRayCenterCrop()(image)
    image = xrv.datasets.XRayResizer(224)(image)
    return torch.from_numpy(image)  # float32, (1, 224, 224)


def load_model():
    manifest = json.loads(MANIFEST.read_text())
    if xrv.__version__ != manifest["torchxrayvision"]:
        raise RuntimeError(f"Use torchxrayvision=={manifest['torchxrayvision']}")
    weights = HERE / manifest["weights"]["path"]
    if not weights.is_file():
        raise FileNotFoundError("Run prepare_xray_demo.py once before the demo")
    if hashlib.sha256(weights.read_bytes()).hexdigest() != manifest["weights"]["sha256"]:
        raise ValueError("Pretrained weights checksum mismatch; run preparation again")
    torch.manual_seed(497050)  # Initial parameters are replaced by pretrained weights.
    model = xrv.models.DenseNet(weights=manifest["model"], cache_dir=str(weights.parent))
    model.eval()
    return model


def sample_paths():
    manifest = json.loads(MANIFEST.read_text())
    paths = []
    for item in manifest["images"]:
        path = HERE / item["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Example image checksum mismatch: {path.name}")
        paths.append(path)
    return paths
