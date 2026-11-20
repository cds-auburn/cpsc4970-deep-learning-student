"""Instructor preparation only. Run from any directory; requires CPU PyTorch.

No tuning against held-out records: fixed seed, architecture, and update count.
"""
import csv
import hashlib
import json

import torch
from heart_model import CONTINUOUS, HERE, LABELS, architecture, prepare

SEED = 497030
FIELDS = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
          "thalach", "exang", "oldpeak", "slope", "ca", "thal", "num"]


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = HERE / "processed.cleveland.data"
    records = []
    for index, row in enumerate(csv.reader(source.open()), start=1):
        original = dict(zip(FIELDS, row))
        # The five selected inputs have no missing values in Cleveland.
        record = {k: float(original[k]) for k in CONTINUOUS}
        record.update(cp=int(float(original["cp"])),
                      target=int(float(original["num"]) > 0), source_row=index)
        records.append(record)
    train, heldout = [], []
    for label in (0, 1):
        indices = [i for i, r in enumerate(records) if r["target"] == label]
        order = torch.randperm(len(indices)).tolist()
        count = round(len(indices) * 0.2)
        heldout.extend(indices[i] for i in order[:count])
        train.extend(indices[i] for i in order[count:])
    numeric = torch.tensor([[records[i][k] for k in CONTINUOUS] for i in train])
    metadata = dict(seed=SEED, continuous=CONTINUOUS, labels=LABELS,
                    mean=numeric.mean(0).tolist(),
                    std=numeric.std(0, correction=0).tolist(),
                    category_order=[1, 2, 3, 4],
                    input_shape=["N", 8], output_shape=["N", 2],
                    train_rows=[records[i]["source_row"] for i in train],
                    heldout_rows=[records[i]["source_row"] for i in heldout],
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    torch_version=str(torch.__version__), updates=500,
                    learning_rate=0.01, weight_decay=0.02)
    x = prepare(records, metadata)
    y = torch.tensor([r["target"] for r in records])
    model = architecture()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.02)
    for _ in range(500):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x[train]), y[train])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        predictions = model(x[heldout]).argmax(1)
    confusion = [[int(((y[heldout] == a) & (predictions == b)).sum())
                  for b in (0, 1)] for a in (0, 1)]
    metadata["heldout_confusion_actual_by_predicted"] = confusion
    metadata["heldout_accuracy"] = float((predictions == y[heldout]).float().mean())
    torch.save(model.state_dict(), HERE / "weights.pt")
    (HERE / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Export every held-out record, in source order: no cherry-picked successes.
    with (HERE / "demo.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["source_row", *CONTINUOUS, "cp", "target"])
        writer.writeheader()
        writer.writerows(records[i] for i in sorted(heldout))
    print(json.dumps({"train": len(train), "heldout": len(heldout),
                      "confusion": confusion, "accuracy": metadata["heldout_accuracy"]}))


if __name__ == "__main__":
    main()
