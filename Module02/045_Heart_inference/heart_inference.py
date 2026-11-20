"""Offline inference; run from any directory with PyTorch installed."""
import csv
import torch
from heart_model import HERE, load_model, prepare


def main():
    model, metadata = load_model()
    with (HERE / "demo.csv").open(newline="") as file:
        records = list(csv.DictReader(file))
    x = prepare(records, metadata)
    with torch.inference_mode():
        scores = model(x)
        probabilities = scores.softmax(dim=1)
        predicted = probabilities.argmax(dim=1)
    print("Inputs:", tuple(x.shape), "Outputs:", tuple(scores.shape))
    print("row  target    prediction  presence score")
    for row, label, p in zip(records, predicted.tolist(), probabilities[:, 1].tolist()):
        print(f"{row['source_row']:>3}  {metadata['labels'][int(row['target'])]:<9} "
              f"{metadata['labels'][label]:<10}  {p:.3f}")
    expected = torch.tensor([int(r["target"]) for r in records])
    print("Correct:", int((predicted == expected).sum()), "/", len(records))


if __name__ == "__main__":
    main()
