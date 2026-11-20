# Classroom Iris classifier

A course-trained demonstration model. It predicts an iris's species (setosa,
versicolor or virginica) from four measurements in centimeters.

## Data and attribution

Fisher, R. A. (1936). [Iris](https://doi.org/10.24432/C56C76). UCI Machine
Learning Repository, [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
The model uses the 150-sample copy distributed with scikit-learn 1.6.1, the same
file M2-030 loads. See [`data/iris/README.md`](../../data/iris/README.md);
`metadata.json` records the source SHA-256.

## Files and supplied helpers

This README provides the instructions for using the model. `metadata.json`
holds values used by the code: feature order, training means and standard
deviations, and class labels. `weights.pt` holds the learned weights and biases.
Keep the weights and metadata together.

[`demos/iris_model.py`](../../demos/iris_model.py) supplies the matching network
architecture and two helpers:

- `load_model()` builds the network, restores the saved weights, selects
  evaluation behavior, and returns `(model, metadata)`.
- `prepare(measurements, metadata)` implements the input preparation described
  below. Pass **unstandardized** measurements with shape `(N, 4)`. The helper
  returns a float32 tensor of the same shape.

[`data/iris/test.csv`](../../data/iris/test.csv) contains the 25 samples kept
out of training, with the recorded species code for comparison. Its `species`
column is not a model input.

## Input preparation

Each flower supplies four continuous measurements, in the order recorded in
`metadata["features"]`:

| Column | Key | Measurement | Unit |
|---|---|---|---|
| 0 | `sepal_length` | Sepal length | cm |
| 1 | `sepal_width` | Sepal width | cm |
| 2 | `petal_length` | Petal length | cm |
| 3 | `petal_width` | Petal width | cm |

1. Put the measurements in a float32 tensor of shape `(N, 4)`, with one row per
   flower. A single flower has shape `(1, 4)`.
2. Standardize each column: `(value - mean) / std`, using the corresponding
   entries in `metadata["mean"]` and `metadata["std"]`. These statistics were
   computed from the 125 training samples, using population standard deviations
   (`correction=0`). Reuse them unchanged; do not recompute statistics from the
   flowers being classified.

`prepare(measurements, metadata)` performs both steps. There are no categorical
inputs and no missing values. Keep the species out of the input.

## Interpreting outputs

`model(x)` returns raw class scores (logits) with shape `(N, 3)`. Each output row
corresponds to the same input flower. The column meanings come from
`metadata["labels"]`, in the order of the species codes in the data files:

| Column | Label |
|---|---|
| 0 | `setosa` |
| 1 | `versicolor` |
| 2 | `virginica` |

The largest score selects the predicted class. Use `scores.argmax(dim=1)` to
obtain one class index per flower, then look up each index in
`metadata["labels"]`.

For nonnegative class scores that sum to one per flower, use
`scores.softmax(dim=1)`. This preserves the winning class; these values have not
been established as calibrated probabilities.

## Run inference

Requires CPU PyTorch and NumPy; the supplied files run offline. From
`Module02`, run this example in Python or a notebook:

```python
import sys
import numpy as np
import torch

sys.path.insert(0, "demos")
from iris_model import load_model, prepare

model, metadata = load_model()
data = np.loadtxt("data/iris/test.csv", delimiter=",", skiprows=1, dtype=np.float32)
test_x = torch.from_numpy(data[:, :4])

x = prepare(test_x[:3], metadata)
model.eval()
with torch.inference_mode():
    scores = model(x)

class_ids = scores.argmax(dim=1)
labels = [metadata["labels"][i] for i in class_ids.tolist()]
print(x.shape, scores.shape)  # torch.Size([3, 4]) torch.Size([3, 3])
print(labels)               # ['virginica', 'setosa', 'virginica']
print(scores.softmax(dim=1))
```

The input helper handles normalization; do not standardize the measurements
before passing them to it. To run the all-sample demo from the repository root:

```sh
python Module02/demos/iris_inference.py
```

## Instructor preparation and evaluation

The supplied weights were trained, and all slide code, the notebook and the
standalone script were executed for M2-030, with Python 3.14.7, PyTorch 2.15.0 and NumPy
2.5.5 on Linux CPU in the course Docker image. Nothing has been executed on
macOS. The demo needs no network or training. It loads a plain tensor
state_dict with `weights_only=True`.

To regenerate the supplied weights, metadata and test CSV, run from the
repository root:

```sh
python Module02/tools/train_iris_model.py
```

Instructor preparation: seed 497040, one CPU thread, deterministic operations.
A random permutation assigns 25 samples to testing and 125 to training; the
split is not stratified, so the test file holds 10 setosa, 9 versicolor and
6 virginica. Metadata saves both row lists, counting from zero in `iris.csv`.
Only training rows determine standardization. Architecture 4 → 8 → 3, ReLU
between linear layers; full-batch Adam, learning rate 0.01, weight decay 0.02,
exactly 500 updates. Settings were fixed before examining test predictions; no
tuning or selection against those results.

All test samples are exported in the order drawn. Accuracy 22/25 (88%);
confusion counts with actual classes as rows and predictions as columns in
setosa/versicolor/virginica order: `[[10, 0, 0], [0, 6, 3], [0, 0, 6]]`. The
three mistakes are versicolor flowers predicted as virginica, with winning
softmax scores of 0.576, 0.615 and 0.569. This single-split result is an
instructor check, not a benchmark. The lecture shows the first three test
samples (rows 72, 16 and 114 of `iris.csv`), and the first is one of the mistakes.

Regeneration across library versions or platforms can change numerical results.
Run slide cells in sequence after the notebook's Setup cell. Training and split
methodology are deferred to later lectures.

## Live notebook

Open [`demos/iris_inference.ipynb`](../../demos/iris_inference.ipynb) in a Jupyter
kernel with PyTorch, NumPy and pandas. Run its Setup cell before class. Two mini-demos load the
150-sample file from GitHub, which needs a network connection; the rest runs
offline. Setup is initially collapsed in notebook
front ends that support the saved metadata; it makes `Module02` the
working directory, so the slide code's relative path works.

Three demonstrations match the lecture: first predict for one flower, then
reconstruct and inspect its four input values, then check the answer and
compare predictions for three flowers. The final optional cell checks all 25
test samples. The recorded species is used only for comparison and never
included in the model input.
