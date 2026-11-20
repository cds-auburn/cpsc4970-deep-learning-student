"""Run from this folder (Module02); each CSV row is one sample."""
import csv
import torch

with open("data/normalization.csv", newline="") as file:
    rows = csv.reader(file)
    next(rows)  # Skip the feature names.
    features = torch.tensor([[float(x) for x in row] for row in rows],
                            dtype=torch.float64)

# Feature matrix X: rows i are samples, columns j are features.
# mean[j] is mu_j; std[j] is sigma_j.
mean = features.mean(dim=0)
std = features.std(dim=0, correction=0)
standardized = (features - mean) / std  # Z_ij for each entry.

print("Original mean:", mean)
print("Original std: ", std)
print("New mean:     ", standardized.mean(dim=0))
print("New std:      ", standardized.std(dim=0, correction=0))

# Undo standardization with the original per-feature statistics.
recovered = standardized * std + mean
print("Recovered original values:", torch.allclose(recovered, features))
