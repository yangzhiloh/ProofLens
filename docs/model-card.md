# ProofLens model card

## Model details and availability

ProofLens is designed as an image-level binary classifier that returns an AI-generated logit, a
normalized feature vector during training, calibrated authentic and AI-generated probabilities,
and inference provenance. The committed backbone is `facebook/dinov2-base` with 224 by 224 RGB
input. A layer-normalization and linear binary head follows the class-token representation.

Primary measured results and distributable weights are unavailable until Task 7 of the
remaining-work plan completes. No checkpoint or ONNX binary is tracked in this repository. Any
fixture metrics produced by `scripts/reproduce_small.py` validate software flow only and are not
primary-dataset results.

## Intended use

The intended use is research and demonstration of image-level authentic versus AI-generated
scoring under documented post-processing conditions. The local app can compare one image with a
deterministically transformed view to expose score instability. Outputs may support controlled
experiments, model comparison, and human review.

This model is not intended to provide forensic proof, determine authorship, make autonomous
moderation decisions, or evaluate video, audio, or pixel-level provenance.

## Architecture and training

The planned experiment sequence is:

1. E0 trains layer normalization and the linear head with the backbone frozen.
2. E1 also fine-tunes the final two DINOv2 transformer blocks.
3. E2 adds clean and randomly transformed classification.
4. E3 adds prediction consistency and normalized-feature consistency.
5. E4 adds loss-guided hard-transformation selection with three candidates.

The committed default loss weights are 1.00 clean binary loss, 1.00 transformed binary loss,
0.25 prediction consistency, and 0.10 feature consistency where the experiment enables them.
Every run records configuration, seed, Git revision, environment versions, manifest and split
hashes, and checkpoint state.

## Datasets

SID-Set and WildFake are primary candidates subject to audits and verified terms. The automated
SID-Set configuration is pinned to revision
`c1674903d858c78e04809c1c6f2703627ac1a621`. WildFake remains manual and has licence status
REQUIRES-VERIFICATION before redistribution. CIFAKE is a separate 32 by 32 low-resolution stress
test and is excluded from primary training. Details are in `docs/datasets.md` and
`THIRD_PARTY_NOTICES.md`.

Labels are `0` for authentic and `1` for AI-generated. Splits operate on source groups and
duplicate clusters, with complete generator families reserved for unseen-generator evaluation.

## Metrics and evaluation

The primary validation score is:

```text
0.50 * clean ROC AUC + 0.50 * macro robust ROC AUC
```

Macro robust ROC AUC first averages severity-level AUC values within each transformation family,
then averages the six family values equally. The evaluation contract also produces clean,
per-condition, per-family, pooled robust, worst-family, worst-condition, and unseen-generator
ROC AUC. Threshold reporting includes accuracy, precision, recall, F1, false positives, and false
negatives after validation-only temperature scaling and threshold selection. Model size,
inference time, error examples, and ONNX parity are also required.

No primary metric value is claimed here. Task 7 must execute E0 through E4 on the frozen manifest
and split, select the validation winner, calibrate it, evaluate untouched test partitions, and
record the resulting evidence before results can be published.

## Limitations

- No primary dataset experiment has yet produced a publishable measured result or weight.
- Performance can shift across cameras, content domains, generators, and post-processing tools.
- Compression, resizing, screenshots, unusual content, and compound transformations can alter a
  score.
- Image-level output does not locate generated regions or establish provenance.
- Calibration cannot repair ranking errors or out-of-domain bias.
- Dataset labels, licensing, source composition, and generator coverage bound what conclusions
  are valid.
- CIFAKE is low resolution and stress-only, so it cannot substitute for primary evaluation.
- OpenVINO is an optional smoke path. CPU ONNX is the required non-CUDA fallback after parity
  passes.

## Ethical considerations

False positives can wrongly characterize authentic work, while false negatives can miss
generated material. Scores require context, uncertainty, and human review. Do not use ProofLens
as the sole basis for accusations, access restrictions, penalties, or rights decisions. Obtain
data lawfully, retain source attribution, avoid publishing restricted images, and inspect errors
across content and source groups before making comparative claims.

The interface explicitly states that scores are not forensic proof. Published reporting must
include false-positive and false-negative examples that can legally be shown, transformation
instability, known failure categories, and measured trade-offs.

## Licences

ProofLens project code is MIT licensed. DINOv2 is recorded as Apache-2.0, SID-Set as CC-BY-4.0
with source-material attribution requirements, and CIFAKE as MIT with required citations.
WildFake is marked REQUIRES-VERIFICATION and must not be redistributed until the terms for the
obtained copy are confirmed. See `THIRD_PARTY_NOTICES.md` for the recorded declarations.
