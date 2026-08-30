# ProofLens Devpost draft

## Publication status

This is a fact-only pre-publication draft. Primary measured results and distributable weights
remain unavailable until Task 7 of the remaining-work plan completes. Calibration, final test
reporting, and parity-gated ONNX are Task 8 outputs. Laptop and clean-checkout acceptance belongs
to Task 9, and the public demonstration recording and publication belong to Task 10. Do not add
performance claims from the miniature fixture workflow.

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
- a Gradio app for clean and transformed probability comparison
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

## Results boundary

No primary clean AUC, robust AUC, worst-condition AUC, unseen-generator AUC, threshold metric,
model size, or inference-time value is asserted in this draft. Task 6 produces E0. Task 7 runs E1
through E4, compares them with E0, and selects the validation winner, making primary measured
results and distributable weights available. Task 8 produces calibration, untouched final test
reporting, error analysis, and a parity-gated ONNX export. Task 9 produces laptop and
clean-checkout acceptance evidence. Only values directly present in those artifacts may be added
to the final entry, and Task 10 performs recording and publication.

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

Publish this entry only in Task 10, after Task 7 supplies measured primary results and weights,
Task 8 supplies calibrated final test and ONNX parity evidence, and Task 9 supplies laptop and
clean-checkout acceptance. The release check must pass on the final tracked tree, public artefact
links must be manually verified, and a human must review the text and record the demonstration
video.
