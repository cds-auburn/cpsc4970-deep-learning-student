"""Pinned, CPU-only loader and the prepared M2-080 example."""

import csv
import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import torch
from chronos import ChronosBoltPipeline

MODULE = Path(__file__).resolve().parents[1]


def load_pipeline():
    """Load verified local weights; preparation is a separate, online step."""
    cache = MODULE / "artifacts/cache/electricity"
    manifest = json.loads((MODULE / "models/electricity/manifest.json").read_text())
    for name in ("config.json", "model.safetensors"):
        path = cache / name
        if not path.exists():
            raise FileNotFoundError("Run tools/prepare_electricity_demo.py first")
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][name]["sha256"]:
            raise ValueError(f"Unexpected checkpoint contents: {name}")
    pipeline = ChronosBoltPipeline.from_pretrained(
        cache, device_map="cpu", dtype=torch.float32, local_files_only=True
    )
    pipeline.model.eval()
    return pipeline


def load_example():
    """Return timestamps and MW observations, with a 168/24 history/future split."""
    with (MODULE / "data/electricity/neso-hourly.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    times = [datetime.fromisoformat(r["hour_start_utc"]) for r in rows]
    if len(times) != 192 or any(b - a != timedelta(hours=1) for a, b in zip(times, times[1:])):
        raise ValueError("Expected eight uninterrupted days of hourly observations")
    demand = torch.tensor([float(r["national_demand_mw"]) for r in rows], dtype=torch.float32)
    if not torch.isfinite(demand).all():
        raise ValueError("The demo requires complete observations")
    return times, demand[:168].unsqueeze(0), demand[168:]
