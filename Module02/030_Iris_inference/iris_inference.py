"""Offline inference; run from any directory with PyTorch and NumPy installed."""
import numpy as np
import torch
from iris_model import HERE, load_model, prepare


def main():
    model, metadata = load_model()
    data = np.loadtxt(HERE / "test.csv", delimiter=",",
                      skiprows=1, dtype=np.float32)
    test_x = torch.from_numpy(data[:, :4])
    test_y = torch.from_numpy(data[:, 4].astype(np.int64))
    x = prepare(test_x, metadata)
    with torch.inference_mode():
        scores = model(x)
        probabilities = scores.softmax(dim=1)
        predicted = probabilities.argmax(dim=1)
    labels = metadata["labels"]
    print("Inputs:", tuple(x.shape), "Outputs:", tuple(scores.shape))
    print("sample  recorded    prediction  winning score")
    for i, (target, label, p) in enumerate(zip(test_y.tolist(), predicted.tolist(),
                                               probabilities.max(dim=1).values.tolist())):
        print(f"{i:>6}  {labels[target]:<10}  {labels[label]:<10}  {p:.3f}")
    print("Correct:", int((predicted == test_y).sum()), "/", len(test_y))


if __name__ == "__main__":
    main()
