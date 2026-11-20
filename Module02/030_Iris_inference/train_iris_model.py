"""Instructor preparation only. Run from any directory; requires CPU PyTorch and NumPy.

No tuning against the test samples: fixed seed, architecture, and update count.
"""
import hashlib
import json

import numpy as np
import torch
from iris_model import FEATURES, HERE, LABELS, architecture, prepare

SEED = 497040
TEST = 25


def main():
    torch.manual_seed(SEED)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    source = HERE / "iris.csv"
    # The same file M2-030 loads from GitHub: one header line, then 150 samples.
    data = np.loadtxt(source, delimiter=",", skiprows=1, dtype=np.float32)
    x_raw = torch.from_numpy(data[:, :4])
    y = torch.from_numpy(data[:, 4].astype(np.int64))
    order = torch.randperm(len(y)).tolist()
    # Keep the shuffled order, so the test file mixes the three species.
    test, train = order[:TEST], order[TEST:]
    metadata = dict(seed=SEED, features=FEATURES, labels=LABELS,
                    mean=x_raw[train].mean(0).tolist(),
                    std=x_raw[train].std(0, correction=0).tolist(),
                    input_shape=["N", 4], output_shape=["N", 3],
                    train_rows=train, test_rows=test,
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    torch_version=str(torch.__version__), updates=500,
                    learning_rate=0.01, weight_decay=0.02)
    x = prepare(x_raw, metadata)
    model = architecture()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.02)
    for _ in range(500):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(x[train]), y[train])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.inference_mode():
        predictions = model(x[test]).argmax(1)
    confusion = [[int(((y[test] == a) & (predictions == b)).sum())
                  for b in range(3)] for a in range(3)]
    metadata["test_confusion_actual_by_predicted"] = confusion
    metadata["test_accuracy"] = float((predictions == y[test]).float().mean())
    torch.save(model.state_dict(), HERE / "weights.pt")
    (HERE / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Export every test sample, in the order drawn: no cherry-picked successes.
    lines = [",".join([*FEATURES, "species"])]
    lines += ["{:.1f},{:.1f},{:.1f},{:.1f},{}".format(*x_raw[i].tolist(), int(y[i]))
              for i in test]
    (HERE / "test.csv").write_text("\n".join(lines) + "\n")
    print(json.dumps({"train": len(train), "test": len(test),
                      "confusion": confusion, "accuracy": metadata["test_accuracy"]}))


if __name__ == "__main__":
    main()
