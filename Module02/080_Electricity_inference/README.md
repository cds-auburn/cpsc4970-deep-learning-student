# Pretrained electricity-demand forecasting demonstration

The demo uses Amazon's [Chronos-Bolt Tiny](https://huggingface.co/amazon/chronos-bolt-tiny)
through the `chronos-forecasting` library, pinned at 2.3.2. `manifest.json` pins
the model revision (`a0e552de`) and the SHA-256 hash of every downloaded file.
Model references: the [model card](https://huggingface.co/amazon/chronos-bolt-tiny),
[Amazon's Chronos-Bolt overview](https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/),
and the [Chronos paper](https://arxiv.org/abs/2403.07815). The model card lists
the weights under Apache 2.0. Data provenance and attribution are under
**Demand data** below.

The model card describes a T5 encoder-decoder with about 9 million parameters,
trained on nearly 100 billion time-series observations. The publisher does not
list individual training series at the level of our excerpt, so **whether these
NESO observations were in the training data is not established**. Treat the
example as an illustration of inference, not a held-out test.

`electricity_model.py` supplies `load_pipeline()`, which verifies the
cached files and returns a CPU `ChronosBoltPipeline` in evaluation mode, and
`load_example()`, which returns the timestamps and the 168/24 history/future
split used by the standalone script.

## Prepare once, run offline

Use the course image, then add the pinned libraries. From this folder:

```sh
python -m pip install -r requirements.txt
python prepare_electricity_demo.py
python electricity_inference.py
```

Verified on Linux x86_64 CPU in the course Docker image: Python 3.14.7,
PyTorch 2.15.0, chronos-forecasting 2.3.2, transformers 5.23.0, NumPy 2.5.5,
Matplotlib 3.11.2. `verified-environment.txt` records every installed package.
The notebook additionally needs Jupyter with `ipykernel` (7.4.0 was used).
macOS and GPU execution have not been checked.

Only preparation needs the network. It downloads the model's `config.json`,
its 35 MB `model.safetensors`, the model card, and NESO's 2024 demand file
(1.8 MB) into ignored `artifacts/cache/`, verifying each hash
against `manifest.json`. It then rewrites the two small tracked CSV excerpts in
this folder. A changed upstream file is an error, never a silent
replacement. `load_pipeline()` loads with `local_files_only=True` and refuses
missing or altered weights rather than downloading.

The standalone script writes `example-results.json` and the five lecture plots
to ignored `artifacts/results/`. The `example-results.json` in this folder
is a saved copy of that output.

For the live demonstration, open
[`electricity_inference.ipynb`](electricity_inference.ipynb) from this folder
and run its Setup cell before presenting. Setup loads `pipeline`; slide code
with relative paths runs as written. The sections follow the slides: Demand alone, Load the model,
The history, Make a forecast, The next day, A range of possibilities, and A
shorter history. Every
code cell has been executed; saved outputs are omitted so students see a fresh
run.

## Input and output contract

**Input.** A float tensor of shape `(N, W)`: `N` histories, each `W` ordered,
equally spaced values of one variable. The Chronos documentation calls a
history the *context*; the lecture and our code say *history*. The demo passes `(1, 168)`: one week of
hourly demand in MW. The model receives values only. It has no timestamps,
units, or other features, so one step means one hour only because the data
were prepared hourly. The checkpoint accepts up to 2,048 context values.

**Scaling.** Pass raw values. Chronos-Bolt standardizes each history with that
history's own mean and standard deviation, and the pipeline converts its
forecasts back to the input's units. The model was trained the same way: its
`forward` applies the same per-history scale to the training targets. So this
is still "prepare the input the way training did", although the tabular model
met that rule with saved training statistics. In `chronos-forecasting` 2.3.2
see `InstanceNorm` in `chronos/chronos_bolt.py`, used by
`ChronosBoltModelForForecasting.encode` and `forward`. Do not standardize the history yourself,
and never include the values being forecast when preparing the input.

**Output.** `pipeline.predict(context, prediction_length=24)` returns shape
`(N, 9, 24)`: history, quantile, future step. The nine rows match
`pipeline.quantiles`, `[0.1, 0.2, ..., 0.9]`; row 4 is the median. All steps
are produced from the same history in one call. The checkpoint was trained for
horizons up to 64 steps; the library warns beyond that.
`ChronosBoltPipeline.predict` runs the model under `torch.no_grad()` (same
file), so no gradient information is kept and the weights do not change;
`forecasts.requires_grad` is `False`. `load_pipeline()` calls `eval()`.

The model expects the value to fall below its 0.9 quantile 9 times in 10 and
below its 0.1 quantile 1 time in 10, so for each hour it expects the value
between indexes 0 and 8 about 8 times in 10. That is the model's estimate of
its uncertainty, not a guarantee for any particular day.

## Recorded example

`example-results.json` holds all nine quantiles for the 24 forecast hours, the
24 observed values, and the environment. For January 15, 2024 (UTC):

- the first three median forecasts are 25,021, 24,690 and 24,415 MW;
- the first hour's median is 220 MW below the observed 25,241 MW;
- the median peaks at 42,092 MW at 17:00 against an observed 45,095.5 MW; and
- the observed value is above the 0.1–0.9 quantile band for the last seven
  hours (17:00–23:00) and inside it for the first seventeen;
- the history's highest hour is 43,580.5 MW, below January 15's observed peak;
  and
- given only the last 24 hours (a Sunday) the median peaks at 38,943 MW.

No accuracy is reported from one day. The run has no sampling or random
augmentation; a rerun in the verified environment reproduced the saved JSON
byte for byte. Values may differ slightly across library versions and
platforms, so compare with a numerical tolerance elsewhere.

## Demand data

Source: [NESO Historic Demand Data 2024](https://www.neso.energy/data-portal/historic-demand-data/historic_demand_data_2024),
published by Great Britain's National Energy System Operator under the
[NESO Open Data Licence](https://www.neso.energy/data-portal/neso-open-licence).

Supported by National Energy SO Open Data.

| File | Contents |
|---|---|
| `neso-half-hourly.csv` | The `ND` (National Demand, MW) column for the 48 settlement periods of each day, January 8–15, 2024: 384 rows. |
| `neso-hourly.csv` | The mean of each hour's two half-hourly values: 192 rows. `hour_start_utc` marks the start of the hour. |

Each half-hourly value covers an equal interval, so their mean is the hour's
average power in MW, not energy. January has no clock change in Great Britain,
so the settlement periods of these eight days align with UTC hours.

The lecture uses the first 168 hourly values (January 8–14, Monday through
Sunday) as the model's history and the last 24 (Monday, January 15) only for
comparison with the forecast.

`prepare_electricity_demo.py` downloads the source file, checks its
SHA-256 hash against `manifest.json`, and regenerates both
excerpts. The complete source file stays in the ignored cache.
