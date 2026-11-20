"""Run the bundled chest X-ray examples offline after one preparation step."""
import argparse
import json
from pathlib import Path

import torch
import torchxrayvision as xrv
from xray_model import load_model, prepare_image, sample_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Save all outputs as JSON")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    paths = sample_paths()
    batch = torch.stack([prepare_image(path) for path in paths])
    model = load_model()
    with torch.inference_mode():
        scores = model(batch)
    print("Input:", tuple(batch.shape), "Output:", tuple(scores.shape))
    results = {"torch": str(torch.__version__), "torchxrayvision": xrv.__version__,
               "labels": model.pathologies, "images": []}
    for path, values in zip(paths, scores):
        named = dict(zip(model.pathologies, values.tolist()))
        print(f"\n{path.name}: five largest finding scores")
        for name, value in sorted(named.items(), key=lambda pair: pair[1], reverse=True)[:5]:
            print(f"  {name:28} {value:.3f}")
        print(f"  Sum over all 18 findings: {values.sum().item():.3f}")
        results["images"].append({"file": path.name, "scores": named})
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
