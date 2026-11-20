# Frozen classroom heart-disease model

A course-trained demonstration, not a clinical prediction system. It classifies
presence/absence labels in historical Cleveland records, not future heart attacks.

## Data and attribution

Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989).
[Heart Disease, UCI](https://doi.org/10.24432/C52P4X),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Downloaded 2026-11-09 from
`https://archive.ics.uci.edu/static/public/45/heart+disease.zip`.
Original documentation and 303-row processed Cleveland file are retained in this
folder (`heart-disease.names`, `processed.cleveland.data`); metadata records the source SHA-256.

## Files and supplied helpers

This README provides the instructions for using the model. `metadata.json`
holds values used by the code: feature order, training means and standard
deviations, and class labels. `weights.pt` holds the learned weights and biases.
Keep the weights and metadata together.

[`heart_model.py`](heart_model.py) supplies the matching network
architecture and two helpers:

- `load_model()` builds the network, restores the saved weights, selects
  evaluation behavior, and returns `(model, metadata)`.
- `prepare(records, metadata)` implements the input preparation described below.
  Pass a list of dictionaries containing the named, **unstandardized** measurements
  and chest-pain codes. The helper returns a float32 tensor of shape `(N, 8)`.

[`demo.csv`](demo.csv) contains 61 records kept out
of training, with recorded labels for comparison. Its `target` and `source_row`
columns are not model inputs.

## Input preparation

Each patient supplies four continuous measurements and one chest-pain category.
Use the following units and continuous-feature order, recorded in
`metadata["continuous"]`:

| Column | Key | Measurement | Unit |
|---|---|---|---|
| 0 | `age` | Age | years |
| 1 | `trestbps` | Resting blood pressure | mm Hg |
| 2 | `chol` | Serum cholesterol | mg/dL |
| 3 | `thalach` | Maximum heart rate achieved | beats/min |

1. Convert these measurements to float32 values in an `(N, 4)` tensor, with one
   row per patient.
2. Standardize each column: `(value - mean) / std`, using the corresponding
   entries in `metadata["mean"]` and `metadata["std"]`. These statistics were
   computed from the 242 training records, using population standard deviations
   (`correction=0`). Reuse them unchanged; do not recompute statistics from the
   patients being classified.
3. Encode `cp` in four one-hot positions ordered 1–4: typical angina, atypical
   angina, non-anginal pain, asymptomatic. Subtract one from the category code
   before `one_hot(..., num_classes=4)`. For example, code 4 becomes `[0, 0, 0, 1]`.
   Convert the indicators to float32 and leave them unstandardized.
4. Append those four indicators after the four standardized measurements.
   The final input has shape `(N, 8)`, including `(1, 8)` for a single patient.

`prepare(records, metadata)` performs all four steps. Supply finite measurements
and integer `cp` codes 1–4; no missing-value imputation is provided. The five
selected features are complete across all 303 Cleveland records; missing values
in other source columns are irrelevant to this model. Keep the target and source
row number out of the input.

## Interpreting outputs

`model(x)` returns raw class scores (logits) with shape `(N, 2)`. Each output row
corresponds to the same input patient. The column meanings come from
`metadata["labels"]`:

| Column | Label | Meaning in these records |
|---|---|---|
| 0 | `absence` | Original diagnosis code `num` was 0 |
| 1 | `presence` | Original diagnosis code `num` was 1, 2, 3, or 4 |

The larger score selects the predicted class. Use `scores.argmax(dim=1)` to
obtain one class index per patient, then look up each index in
`metadata["labels"]`. The recorded binary target in `demo.csv` is `int(num > 0)`
and is used only to check predictions.

For nonnegative class scores that sum to one per patient, use
`scores.softmax(dim=1)`. This preserves the winning class; these values have not
been established as calibrated clinical probabilities. The task is disease
presence in historical records, not future heart-attack risk.

## Run inference

Requires CPU PyTorch; the supplied files run offline. From this folder,
run this example in Python or a notebook:

```python
import csv
import sys
import torch

sys.path.insert(0, "demos")
from heart_model import load_model, prepare

model, metadata = load_model()
with open("demo.csv", newline="") as file:
    records = list(csv.DictReader(file))

x = prepare(records[:3], metadata)
model.eval()
with torch.inference_mode():
    scores = model(x)

class_ids = scores.argmax(dim=1)
labels = [metadata["labels"][i] for i in class_ids.tolist()]
print(x.shape, scores.shape)  # torch.Size([3, 8]) torch.Size([3, 2])
print(labels)               # ['presence', 'presence', 'absence']
print(scores.softmax(dim=1))
```

The input helper handles normalization; do not standardize the records before
passing them to it. To run the existing all-record demo from this folder:

```sh
python heart_inference.py
```

## Instructor preparation and evaluation

Requires CPU PyTorch. Verified with Python 3.12.15 and PyTorch 2.2.2 on macOS.
On 2026-11-18 the M2-045 slide code and every notebook cell were also executed
with Python 3.14.7 and PyTorch 2.15.0 on Linux CPU in the course Docker image;
the saved weights reproduced the slide values to three decimals.
The demo needs only PyTorch and Python's standard library, with no network or
training. It loads a plain tensor state_dict with `weights_only=True`.

To regenerate the supplied weights, metadata and demo CSV, run from this folder:

```sh
python train_heart_model.py
```

Instructor preparation: seed 497030, one CPU thread, deterministic operations.
A stratified random split assigns 242 records to training and 61 to the demo;
metadata saves source-row lists. Only training rows determine standardization.
Architecture 8 → 12 → 2, ReLU between linear layers; full-batch Adam, learning
rate 0.01, weight decay 0.02, exactly 500 updates. Settings were fixed before
examining held-out predictions; no tuning or selection against those results.

All held-out rows are exported in source order. Accuracy 41/61 (67.2%); confusion
counts with actual classes as rows and predictions as columns in absence/presence
order: `[[22, 11], [9, 19]]`. This modest single-split result is an instructor
check, not clinical validation or a benchmark. The lecture shows source rows
2, 11, 12, the first three demo records, including a mistake.

Keep weights and metadata together. Regeneration across library versions or
platforms can change numerical results. The supplied `heart_model.py` defines
the matching architecture and equivalent preprocessing shown in the slides.
Run slide cells in sequence after the notebook's Setup and Load the model cells.
Training and split methodology are deferred to later lectures.

## Live notebook

Open [`heart_inference.ipynb`](heart_inference.ipynb) in a Jupyter
kernel with PyTorch and pandas. Run its Setup and Load the model cells before
presenting; they load the CSV records, frozen `model` and saved `metadata`.
Setup is initially collapsed
in notebook frontends that support the saved metadata. The slide snippets
continue that session. Open the notebook from this folder.

The lecture is M2-045. Three
demonstrations match it: first predict for one patient,
then reconstruct and inspect all eight input values, then try another patient
and compare predictions for three records. The middle section shows both raw
scores and their softmax transformation before decoding the label. The final
optional cell checks all 61 demo records. The recorded target is used only for
comparison and never included in the model input.

Every notebook cell and the slide snippets have been executed; reconstruction
matches the shared helper and the predictions reproduce the existing frozen
model's results. The standalone script remains the simple all-record terminal
demo and needs neither pandas nor notebook support.
