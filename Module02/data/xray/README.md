# Chest X-ray demonstration images

Data provider: **NIH Clinical Center**. Source collection:
[NIH ChestX-ray14 download site](https://nihcc.app.box.com/v/ChestXray-NIHCC).

Citation: Wang, X., Peng, Y., Lu, L., Lu, Z., Bagheri, M., & Summers, R. (2017).
*ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised
Classification and Localization of Common Thorax Diseases*. IEEE CVPR, 3462–3471.

These two 512×512, 8-bit grayscale PNG examples are the files distributed in the
[TorchXRayVision test collection](https://github.com/mlmed/torchxrayvision/tree/17726dd8f5e756eec3e5ae958281a289b05b1dae/tests):
`00000001_000.png` and `00027426_000.png`. They are retained byte-for-byte;
URLs and SHA-256 hashes appear in `../../models/xray/manifest.json`.
Retrieved 2026-11-09. Selection: the two NIH-named PNGs in that upstream folder,
not examples selected for favorable model predictions.

[NIH image use and attribution](https://docs.cloud.google.com/healthcare-api/docs/resources/public-datasets/nih-chest#license_and_attribution)
permits unrestricted image use with the NIH download link, paper citation and
provider acknowledgement above. This applies to the NIH images; it does not
assert that every image in the upstream repository has the same terms.

No reference diagnosis annotations are bundled for these images. The upstream
collection calls them tests, but their membership in this model's training data
is unknown: do not use this pair to claim held-out accuracy or independent
clinical validation. The lecture's scores are model outputs, not ground truth.
