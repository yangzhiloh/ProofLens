# ProofLens Devpost draft

## Publication status

This is the fact-checked publication copy. E0 through E4, validation selection, calibration,
untouched-test reporting, parity-gated ONNX export, CPU and clean-checkout acceptance, and
separately hosted model artifacts are complete. The repository and release destinations are
listed below. A human still needs to add the verified demonstration URL and submit this copy
through the publication account. Fixture workflow values are not project performance claims.

- Repository: https://github.com/yangzhiloh/ProofLens
- Model release: https://github.com/yangzhiloh/ProofLens/releases/tag/prooflens-e2-rc1
- Demonstration video: add the verified public URL after upload

The linked RC1 release contains the earlier public demo bundle. Publish the matching E2-on-SID
bundle before submitting the latest evaluation results below.

## Project summary

ProofLens is a research prototype for authentic versus AI-generated image classification that
tests whether a detector's predictions survive realistic post-processing. The committed
evaluation covers JPEG compression, Gaussian blur, downscale and upscale, Gaussian noise, color
jitter, and center crop. Separate generator-family partitions are reserved to measure transfer
beyond generators used for training.

## The problem

Image detectors can learn shortcuts that disappear after ordinary redistribution. Compression,
resizing, and editing may alter the evidence on which a clean-image classifier relies. A useful
experiment therefore needs condition-specific robustness measurement, data leakage controls,
and a local inference path that uses exactly the selected and calibrated model.

## What was built

ProofLens includes:

- dataset adapters, canonical manifests, audits, checksums, perceptual hashes, and grouped splits
- deterministic canonical transforms shared by training and evaluation
- a DINOv2 base detector with staged head and final-two-block training
- clean and transformed classification, prediction consistency, feature consistency, and
  loss-guided hard-transform selection
- clean, robust, worst-condition, and unseen-generator evaluation
- validation-only temperature calibration and deterministic checkpoint selection
- PyTorch inference, parity-gated ONNX export, and optional OpenVINO smoke testing
- a Gradio app that scores already processed images exactly as uploaded
- recursive directory inference with portable `image_path` and calibrated `pred` JSON records
- machine-readable metrics, robustness tables, plots, and error-gallery support
- an offline miniature workflow plus Windows and Linux CPU CI

## Technical approach

Each source image receives a canonical manifest row with label, dataset and generator metadata,
source-group identity, dimensions, format, licence identifier, and hashes. Splitting keeps source
groups and duplicate clusters together and holds out complete generator families.

The model uses `facebook/dinov2-base`, layer normalization, and a linear binary head. E0 trains
the head, E1 fine-tunes the final two transformer blocks, E2 adds transformed classification, E3
adds prediction and feature consistency, and E4 adds three-candidate hard-transform selection.
The validation score weights clean ROC AUC and family-macro robust ROC AUC equally.

Calibration is fitted only after checkpoint selection. ONNX is published only if 32-sample
PyTorch and ONNX parity meets the configured tolerance. The Gradio app then uses the same
calibrated inference service as programmatic inference.

## Reproducibility

The repository records experiment configurations for E0 through E4. Run metadata captures the
Git revision, environment, backbone identifier, seed, manifest hash, split hash, and
configuration hash. A one-command fixture run exercises manifest creation, grouped splitting,
training, evaluation, calibration, and reporting without downloads:

```text
uv run --locked --extra dev python scripts/reproduce_small.py --output artifacts/release-smoke --experiment e4
```

The fixture run proves that the software path executes. It does not estimate primary-model
accuracy or robustness.

## Results

E2-on-SID was selected using validation data. After validation-only calibration, the untouched
primary test set of 1,006 images produced 0.9974 clean ROC AUC, 0.9971 family-macro robust ROC
AUC, 0.9966 worst-condition ROC AUC, and 0.9638 unseen-generator ROC AUC. At the selected
threshold, accuracy was 0.9841, precision 0.7846, recall 0.9623, and F1 0.8644, with 14 false
positives and 2 false negatives. A 32-sample ONNX comparison passed the `1e-4` parity tolerance.

On a Windows CPU, the accepted production model loaded successfully and one image prediction
used 323.58 ms of model inference. This is one acceptance observation, not a benchmark
distribution. The displayed app now analyzes each uploaded image as received. Robustness is
supported by the condition-level test results above rather than an interactive comparison. Full
evidence and limitations are in `docs/reports/` and `docs/model-card.md`.

## Limitations and responsible use

ProofLens is an image-level research demonstration, not forensic proof. Domain shifts,
unrepresented generators, unusual content, screenshots, and compound transformations may change
scores. False positives and false negatives can cause harm if treated as conclusions. The system
must remain one input to contextual human review and must not independently determine authorship,
moderation, access, or penalties.

Dataset and model terms remain separate from the MIT project code. WildFake is explicitly marked
REQUIRES-VERIFICATION before redistribution. Raw data, checkpoints, predictions, and ONNX files
are excluded from Git and require separate publication review.

## Final publication gate

The final tracked tree and evaluation evidence have automated verification in
`docs/reports/publication-checklist.md`. Before submitting, publish the matching E2-on-SID model
bundle, insert the verified demonstration URL above, review this copy, and confirm the destination
platform's terms and required fields.
