# Pretrained chest X-ray inference demonstration

The demo uses [TorchXRayVision](https://github.com/mlmed/torchxrayvision), pinned
at 1.5.7, with the `densenet121-res224-all` DenseNet-121 checkpoint.
Model references: [library paper](https://arxiv.org/abs/2111.00595) and
[checkpoint study](https://arxiv.org/abs/2002.02497).
Upstream code licensing: [Apache 2.0](https://github.com/mlmed/torchxrayvision/blob/main/LICENSE).
Image provenance and attribution are in `../../data/xray/README.md`.

The checkpoint's upstream metadata lists the training sources as
`nih-pc-chex-mimic_ch-google-openi-rsna`: NIH ChestX-ray14, PadChest,
CheXpert, MIMIC-CXR with CheXpert labels, Google-labeled NIH images, OpenI,
and the RSNA Pneumonia Challenge. The [model documentation](https://mlmed.org/torchxrayvision/models.html)
identifies the available weights; the checkpoint study above describes the
cross-dataset work. These are the training sources, not a claim that our two
demo images were held out.

For the public model's usage example, see the
[upstream README](https://github.com/mlmed/torchxrayvision#getting-started).
The **Preprocessing and output contract** section below records the exact
requirements for our pinned version and bundled files. `demos/xray_model.py`
supplies `load_model()` and `prepare_image(path)`; the loaded model supplies
its ordered output names through `model.pathologies`.

## Prepare once, run offline

Use a CPU environment with a compatible PyTorch/torchvision pair (the course
image already supplies those). Add the pinned library:

```sh
python -m pip install -r Module02/demos/xray-requirements.txt
python Module02/tools/prepare_xray_demo.py
python Module02/demos/xray_inference.py
```

Commands above run from the repository root. Python 3.12.15, PyTorch 2.2.2,
torchvision 0.17.2, NumPy 1.26.4 and TorchXRayVision 1.5.7 were verified on
macOS CPU; `verified-environment.txt` records the full installed environment.
This is a verification record, not a recommendation to downgrade the course
image. Newer PyTorch and the course Linux container remain to be checked.
No GPU, dataset registration or clinical-data account is required.

Only preparation needs the network. It downloads two small public PNGs and
one approximately 28 MB checkpoint, verifying their SHA-256 hashes against
`manifest.json`. Images are tracked with the course; the checkpoint is cached
under ignored `artifacts/xray/weights/` rather than committed to Git. On another
computer, rerun preparation. It verifies existing files and skips downloading
them. Delete a corrupted cached checkpoint and rerun preparation if necessary.
Downloads use a temporary file so interruptions do not leave a partial cache.

The standalone demo refuses missing/corrupted weights rather than downloading
silently. TorchXRayVision's underlying constructor can download missing weights;
the supplied helper checks the cache first. Upstream 1.5.7 loads a
serialized model with `weights_only=False`; use only the exact official file
whose checksum is pinned in the manifest.

Run the notebook's Setup cell before presenting; this loads `model` and locates
the example files. Slide snippets continue that prepared session. To run slide
cells using their relative paths, use `Module02` as the working
directory. The standalone script resolves paths independently of its working
directory. To save every output:

```sh
python Module02/demos/xray_inference.py --output Module02/artifacts/xray/results.json
```

For the live visual demonstration, open
[`demos/xray_inference.ipynb`](../../demos/xray_inference.ipynb) in Jupyter
using this environment's kernel. Its three demonstrations follow the revised
slides: make a first prediction, inspect/reconstruct that input, then compare
another image. Setup is separate and initially collapsed in supporting notebook
frontends. Each image remains associated with its results. The main comparison
uses two named findings (Cardiomegaly and Nodule) with plain definitions; all 18
scores are available in an optional final cell.
Jupyter supplies IPython for image/table display. Every code cell has been
executed and the notebook structure validated; saved cell outputs are omitted
so students can inspect a fresh run.

## Preprocessing and output contract

The bundled files are Pillow mode `L`: 8-bit grayscale, initially 512×512.
Reject other modes rather than silently truncating 16-bit or DICOM images.
Use the upstream transforms in this order:

1. `xrv.datasets.normalize(pixels, 255)` maps 0–255 to -1024–1024.
2. Add the one-channel dimension to make `(1, H, W)`.
3. `XRayCenterCrop()` crops the long dimension to a square.
4. `XRayResizer(224)` resamples with its default scikit-image engine to float32.
5. Stack images into `(N, 1, 224, 224)`; run an eval-mode model under
   `torch.inference_mode()` on CPU.

The examples are already square, so the crop is a no-op for them. Resizing
loses spatial detail; keep the original files for display. This input scaling
is a fixed-range transformation, not the earlier lecture's mean/std
standardization. Never standardize each image independently or use RGB/ImageNet
preprocessing with this checkpoint.

The output is `(N, 18)` aligned with `model.pathologies`. In this pinned
configuration, the library applies sigmoid followed by per-finding operating
point rescaling (`op_norm`) internally: **these are already bounded scores,
not raw logits**. Do not apply another sigmoid or softmax. The 18 findings can
coexist, so outputs are not a distribution over mutually exclusive categories
and need not sum to one. In particular, 0.5 is not an established 50% clinical
risk, and neither top-k ranking nor a universal cutoff establishes a diagnosis.

`example-results.json` contains the actual outputs for both examples and the
18 labels in model order. The first image's largest score is Cardiomegaly
(about 0.622); the second image's largest score is Mass (about 0.606). These are
predictions, not verified diagnoses. No accuracy is reported, since reference
labels and independence from training are not established for this pair.

## Reproducibility

Checkpoint and image hashes are pinned. There are no random augmentations.
The demo uses one CPU thread and deterministic algorithms, and model creation
uses seed 497050 before replacing initialized weights with pretrained values.
Exact output values can vary across library versions and platforms; compare
with a numerical tolerance rather than relying on bitwise cross-platform identity.
No training or parameter updates occur in the inference demo.
