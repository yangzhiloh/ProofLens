# Robust AI Image Detector: System Design Specification

Date: 2026-08-28  
Status: Approved design, pending written-specification review  
Primary owner: Codex  
Target: Four-day hackathon prototype

## 1. Purpose

Build a reproducible image-level binary classifier that distinguishes authentic images from AI-generated images while remaining accurate after realistic post-processing and redistribution transformations.

The project must prioritize three properties:

1. Robustness to JPEG compression, blur, resizing, noise, color jitter, and center cropping.
2. Generalization to generator families not used for training.
3. A complete and reproducible submission containing code, model weights, evaluation reports, error analysis, documentation, and a local web demonstration.

The primary technical contribution is survival-aware training. Each image is paired with a transformed view, the detector is trained to keep its predictions and representations consistent across the pair, and a loss-guided curriculum selects transformations that expose the detector's current weaknesses.

## 2. Scope

### 2.1 Included

1. Image-level binary classification.
2. Public or properly licensed datasets.
3. One public pretrained vision backbone.
4. Leakage-safe dataset manifests and splits.
5. Deterministic robustness evaluation.
6. Paired clean and transformed training.
7. Hard-transformation selection.
8. Cross-generator evaluation.
9. Probability calibration.
10. PyTorch inference and ONNX export.
11. Attempted OpenVINO conversion, with CPU ONNX inference as the required fallback.
12. A local image-upload web interface.
13. Automated report and error-gallery generation.
14. Reproducible commands, tests, and documentation.

### 2.2 Excluded from the primary build

1. Video or audio detection.
2. Pixel-level localization.
3. Production deployment or moderation integration.
4. Mandatory test-time augmentation.
5. Multi-model ensembles.
6. Dependence on image metadata at inference.
7. Training a vision backbone from scratch.
8. Newly generated synthetic content unless dataset rules and licensing clearly allow it.
9. A frequency branch or authentic-manifold branch without supporting ablation evidence.

## 3. Competition and operating constraints

1. The complete model must contain fewer than 2 billion parameters.
2. Pretrained backbones must be publicly available.
3. Custom project code will be released under an MIT or Apache-compatible licence.
4. Training and evaluation data must be public or properly licensed.
5. Derived transformation scripts must be reproducible.
6. The public repository must contain a clear README and run commands.
7. The final deliverables include a written description, public code, model weights, robustness table, false-positive and false-negative analysis, and a two-to-four-minute public demonstration video.
8. The development window is four days.
9. Training may use stronger external hardware, but inference must run on the target Windows laptop.
10. The target laptop has approximately 32 GB of system RAM, an integrated Intel Arc 140T GPU using shared memory, and an Intel AI Boost NPU. CUDA is unavailable.
11. No strict inference-latency target applies. Reliability and accuracy take priority.

## 4. Evaluation assumptions

The webinar did not define the precise aggregation of robust ROC AUC. The project will use a conservative local contract and report multiple interpretations.

### 4.1 Primary checkpoint score

```text
project_score = 0.50 * AUC_clean + 0.50 * AUC_robust_macro
```

`AUC_robust_macro` is calculated in two stages:

1. Average severity-level AUC values within each transformation family.
2. Average the six transformation-family AUC values with equal weight.

This prevents a family with more listed severities from receiving disproportionate weight.

### 4.2 Additional reported metrics

1. Clean ROC AUC.
2. AUC for every transformation and severity.
3. Macro AUC for every transformation family.
4. Pooled robust ROC AUC across all transformed samples.
5. Worst-family ROC AUC.
6. Worst-condition ROC AUC.
7. Unseen-generator ROC AUC.
8. Accuracy, precision, recall, and F1 at a validation-selected threshold.
9. False-positive and false-negative counts.
10. Model size and inference time.

Checkpoint ties are resolved using worst-family AUC, followed by unseen-generator AUC, followed by lower system complexity.

## 5. System architecture

```text
Public datasets
    -> dataset adapters and licence registry
    -> canonical manifest
    -> deduplication and grouped splitting
    -> clean and transformed training pairs
    -> survival-aware detector training
    -> calibration and checkpoint selection
    -> clean, robust, and unseen-generator evaluation
    -> PyTorch and ONNX export
    -> local web interface and reproducible reports
```

### 5.1 Subsystem boundaries

1. **Dataset subsystem**: Acquires or imports datasets, validates expected files, records licences, audits distributions, creates manifests, finds duplicates, and creates grouped splits.
2. **Transformation subsystem**: Implements canonical evaluation transformations and randomized training transformations through a shared interface.
3. **Model subsystem**: Wraps the public backbone, classification head, normalized representation, loss components, and checkpoint loading.
4. **Training subsystem**: Creates paired batches, executes baseline and robust training, performs hard-transformation selection, saves resumable checkpoints, and records experiments.
5. **Evaluation subsystem**: Runs clean, robust, cross-generator, calibration, latency, and export-parity evaluation.
6. **Reporting subsystem**: Produces machine-readable metrics, Markdown tables, plots, robustness summaries, and error galleries.
7. **Inference subsystem**: Applies canonical preprocessing and returns authentic probability, AI-generated probability, and calibrated confidence.
8. **Export subsystem**: Exports ONNX, validates numerical parity, and attempts OpenVINO conversion.
9. **Web subsystem**: Provides local upload, inference, transformation comparison, score stability, limitations, and downloadable result views.

Every subsystem communicates through explicit configuration objects, manifest rows, checkpoint metadata, or result files. Dataset-specific behavior must not leak into the model or evaluation code.

## 6. Repository structure

```text
configs/
    data/
    experiments/
    export/
docs/
    superpowers/specs/
    reports/
scripts/
    acquire_data.*
    build_manifest.*
    train.*
    evaluate.*
    export.*
    run_demo.*
src/
    data/
        adapters/
        manifest/
        splitting/
        transforms/
    models/
    training/
    evaluation/
    reporting/
    inference/
    export/
    web/
tests/
    fixtures/
    unit/
    integration/
artifacts/
    manifests/
    splits/
    checkpoints/
    predictions/
    reports/
```

Large datasets and generated artifacts will be excluded from version control. The repository will contain configuration, schemas, small test fixtures, checksums, and acquisition instructions.

## 7. Data design

### 7.1 Initial datasets

1. WildFake and SID_Set are primary candidates, subject to licence and structure verification.
2. CIFAKE is treated as a separate 32-by-32 low-resolution stress test.
3. No dataset is merged into primary training until its class balance, resolution, format, content, source, and generator distributions have been audited.

### 7.2 Canonical manifest

Every source image receives one manifest row containing:

```text
sample_id
path
label
dataset_name
dataset_version
generator_family
source_group_id
original_image_id
width
height
file_format
licence_identifier
content_checksum
perceptual_hash
split
```

Labels use `0` for authentic and `1` for AI-generated.

### 7.3 Dataset adapters

Each adapter must:

1. Validate expected files and directory structure.
2. Record dataset version and licence information.
3. Normalize labels and generator metadata.
4. Generate canonical manifest rows.
5. Report malformed, unsupported, or missing images.
6. Avoid silently assigning uncertain labels.

Download automation is used only when terms permit it. Otherwise, the adapter reports the manual download and placement steps.

### 7.4 Decoding and normalization

1. Preserve original files for audit and forensic inspection.
2. Apply EXIF orientation before model preprocessing.
3. Decode every valid image to RGB.
4. Record original dimensions and format before resizing.
5. Log corrupt images and skip them unless corruption exceeds a configured safety limit.
6. Apply model-specific resizing only inside the data loader or inference preprocessor.

### 7.5 Leakage-safe splitting

1. Split source groups rather than files.
2. Keep every original image and all descendants in one partition.
3. Deduplicate across datasets before split assignment.
4. Preserve reliable official splits where compatible with the evaluation design.
5. Hold out complete generator families for cross-generator evaluation.
6. Preserve both labels in every evaluation partition and stratify labels subject to source-group integrity.
7. Keep calibration, checkpoint selection, and thresholds independent from final test partitions.

Manifest creation must fail if it detects cross-split checksum or source-group leakage.

## 8. Transformation design

### 8.1 Canonical evaluation transformations

1. **JPEG compression**: RGB JPEG round-trip at qualities 90, 70, 50, and 30, using a recorded codec implementation and 4:2:0 chroma subsampling.
2. **Gaussian blur**: Sigma 0.5, 1.0, and 2.0, with an odd kernel covering approximately three sigma on each side.
3. **Resize**: Downsample each image dimension to 0.5 or 0.25 of its original size, then upscale to the original size using bicubic interpolation.
4. **Gaussian noise**: Add independent noise at sigma 0.02, 0.05, and 0.10 to pixels scaled to `[0,1]`, then clip to `[0,1]`.
5. **Color jitter**: Sample brightness, contrast, and saturation factors from `[0.8,1.2]`. Hue jitter is excluded because the webinar did not specify it.
6. **Center crop**: Retain the center 80 percent of image width and height, then resize to the original dimensions using bicubic interpolation.

Primary evaluation applies one transformation at a time. Compound transformations and re-screenshot simulation are secondary stress tests and do not drive primary checkpoint selection.

### 8.2 Training transformations

1. Sample transformation families uniformly.
2. Sample listed severities uniformly within a selected family.
3. Randomize implementation details such as interpolation where this improves library robustness without changing the semantic transformation.
4. Preserve clean examples in every paired batch.
5. Record random seeds and transformation parameters for debugging.

Evaluation transformations are deterministic and cacheable. Training transformations are generated on demand.

## 9. Model design

### 9.1 Primary backbone

The primary backbone is DINOv2-B/14 with 224-by-224 RGB input. CLIP ViT-B is not part of the committed experiment sequence and can be evaluated only after the DINOv2 pipeline and required deliverables are complete.

### 9.2 Classification head

The DINOv2 768-dimensional class-token representation feeds layer normalization followed by one linear binary-classification layer. The model returns:

1. An AI-generated logit.
2. A normalized feature vector for consistency training.

The initial head is deliberately small so observed improvements can be attributed to robust training rather than excessive classifier capacity.

### 9.3 Training stages

1. Frozen backbone with the layer-normalization and linear classification head.
2. Fine-tuning of the final two DINOv2 transformer blocks together with the classification head.
3. Clean and randomly transformed classification.
4. Prediction and feature consistency.
5. Loss-guided hard-transformation selection.

Each stage must demonstrate an improvement under the shared evaluation contract before becoming part of the final model.

### 9.4 Loss design

```text
total_loss =
    1.00 * clean_binary_loss
  + 1.00 * transformed_binary_loss
  + 0.25 * prediction_consistency_loss
  + 0.10 * feature_consistency_loss
```

These coefficients are initial configuration defaults. Experiments may change them, but every change must be recorded and evaluated against the same split.

Prediction consistency is mean-squared error between the clean and transformed logits. Feature consistency is one minus cosine similarity between the normalized clean and transformed representations.

### 9.5 Hard-transformation selection

For each selected training image, the default candidate count is three:

1. Sample three transformation families without replacement and sample one listed severity from each family.
2. Estimate their effect on the correct classification margin.
3. Select the candidate that most reduces the margin.
4. Train against the selected candidate together with the clean view.
5. Preserve uniform family coverage so one difficult family cannot monopolize training.

The candidate count remains configuration-controlled, but three is the committed default and any change is treated as an experiment.

### 9.6 Optional ablation-gated modules

1. A frequency or residual branch may be implemented only after the primary model and report pipeline are complete.
2. A real-only prototype or authentic-manifold head may be implemented only if exploratory embeddings show stable and useful separation.
3. An optional module enters the final system only if it improves the composite validation score, does not materially damage worst-family performance, exports successfully, and remains runnable on the target laptop.

## 10. Training and experiment management

1. Every run uses a versioned configuration file.
2. Every run records git revision, environment versions, dataset manifest hash, split hash, seed, hyperparameters, and transformation policy.
3. Checkpoints include model, optimizer, scheduler, epoch, global step, random-number states, and configuration.
4. Training supports deterministic resume after interruption.
5. Out-of-memory recovery may reduce batch size and increase gradient accumulation, but it must record the effective change.
6. The selected backbone or objective must never change silently.

### 10.1 Experiment sequence

| Experiment | Purpose | Advancement rule |
| --- | --- | --- |
| E0 | Frozen backbone plus linear head | Establish a functioning baseline |
| E1 | Fine-tune the final two DINOv2 transformer blocks | Keep only if the composite score improves |
| E2 | Clean and randomly transformed classification | Measure standard augmentation value |
| E3 | Add prediction and feature consistency | Keep only if robust gain justifies clean-score change |
| E4 | Add hard-transformation selection | Primary innovation candidate |
| E5 | Optional frequency branch | Attempt only after the core system is complete |
| E6 | Optional authentic prototype head | Attempt only with supporting embedding evidence |

All experiments use compatible splits, metrics, and seeds.

## 11. Calibration and inference

1. Select the final checkpoint before fitting calibration.
2. Fit temperature scaling using validation predictions only.
3. Official inference uses one image and one forward pass by default.
4. Return authentic probability, AI-generated probability, calibrated confidence, model version, and preprocessing version.
5. Reject unsupported or corrupt inputs with a readable error.

Calibration does not alter ROC ranking but improves confidence displays and threshold-based error analysis.

## 12. Export and laptop compatibility

1. PyTorch is the reference inference implementation.
2. Export the final selected model to ONNX.
3. Compare ONNX and PyTorch predictions on a fixed parity set.
4. Reject the export if numerical disagreement exceeds the configured tolerance.
5. Attempt OpenVINO conversion for the Intel target laptop.
6. Retain CPU ONNX inference as the guaranteed non-CUDA fallback.
7. Do not make Intel NPU acceleration a completion requirement.

## 13. Local web interface

The local interface uses Gradio and must:

1. Accept common image formats by upload.
2. Display authentic and AI-generated probabilities.
3. Display calibrated confidence and a non-forensic disclaimer.
4. Allow the user to apply one supported transformation and compare prediction stability.
5. Show the clean and transformed images side by side.
6. Show model version, selected transformation, parameters, and inference time.
7. Present concise limitations and known failure categories.
8. Handle invalid uploads without stopping the application.

The interface is a demonstration layer over the shared inference API. It must not contain separate model logic.

## 14. Reporting and submission assets

The report generator produces:

1. JSON and CSV metrics.
2. A Markdown robustness table.
3. Per-family and per-severity plots.
4. Clean-versus-robust comparison plots.
5. Unseen-generator results.
6. Highest-confidence false-positive examples.
7. Highest-confidence false-negative examples.
8. Transformation-instability examples.
9. A model and inference summary.
10. Assets suitable for the README, Devpost entry, and video.

Reports must distinguish measured results, assumptions, and limitations.

## 15. Failure handling

1. Missing datasets produce placement and acquisition guidance.
2. Corrupt images are logged and skipped within a configured safety threshold.
3. Duplicate or split leakage stops manifest creation.
4. Unsupported single-class evaluation partitions fail with a clear metric error.
5. Interrupted training resumes from a complete checkpoint.
6. Out-of-memory recovery records every effective configuration change.
7. Export parity failure prevents publishing the exported artifact.
8. Invalid web uploads return readable errors without terminating the server.
9. Missing optional acceleration falls back to CPU inference.

## 16. Testing design

### 16.1 Unit tests

1. Canonical transformation severities and dimensions.
2. Deterministic evaluation transformations.
3. Manifest parsing and label normalization.
4. Checksums and perceptual hashes.
5. Grouped split and leakage detection.
6. Metric calculations using known examples.
7. Loss values and gradient propagation.
8. Checkpoint save and resume.
9. Calibration behavior.

### 16.2 Integration tests

1. Build a manifest from a miniature fixture dataset.
2. Train for one short epoch.
3. Evaluate clean and transformed partitions.
4. Generate a robustness report and error gallery.
5. Export the miniature model to ONNX.
6. Compare PyTorch and ONNX predictions.
7. Launch the web application and complete one inference request.

### 16.3 Reproducibility test

A clean repository checkout must reproduce a small end-to-end workflow by following the README. This test is required before submission.

## 17. Ownership and external dependencies

Codex owns the complete technical project:

1. Repository and system architecture.
2. Dataset preparation code and audits.
3. Model, training, evaluation, reporting, export, and web code.
4. Experiment execution where the environment permits.
5. Tests, documentation, technical write-up, and video script.
6. Final integration and reproducibility checks.

Human assistance is limited to operations Codex cannot independently complete:

1. Accepting licences or signing into external services.
2. Starting cloud jobs when interactive credentials are required.
3. Recording and uploading the public video.
4. Reviewing and approving publication.
5. Supplying decisions when external rules or results require human judgment.

The design must remain locally runnable even when external training hardware is temporarily unavailable.

## 18. Four-day delivery sequence

### Day 1: Foundation and baseline

1. Create the repository structure and environment.
2. Implement dataset adapters, manifests, audits, splits, and transforms.
3. Implement the model wrapper, baseline training, evaluation foundation, and tests.
4. Acquire usable primary data or prepare precise manual acquisition instructions.
5. Train and evaluate E0. Start E1 immediately after E0 produces a valid checkpoint and complete metric report.

Day 1 exit condition: A baseline trains, saves, reloads, evaluates, and emits continuous predictions.

### Day 2: Robust training

1. Complete canonical robustness evaluation.
2. Run E2 and E3.
3. Implement and run E4.
4. Generate the first complete robustness table.
5. Inspect source bias and transformation-specific failures.

Day 2 exit condition: At least one robust model improves the validation composite over E0 without unacceptable clean-score degradation.

### Day 3: Selection, export, and demonstration

1. Repeat the strongest configuration with additional seeds if compute permits.
2. Run cross-generator evaluation.
3. Decide whether optional E5 or E6 is justified.
4. Select and calibrate the final checkpoint.
5. Export and validate ONNX, then attempt OpenVINO conversion.
6. Build and test the local web interface.
7. Generate initial plots and error galleries.

Day 3 exit condition: The final candidate runs on the target laptop and produces complete evaluation artifacts.

### Day 4: Submission hardening

1. Freeze the selected model and environment.
2. Run final clean, robust, stress, and unseen-generator evaluations.
3. Complete a clean-checkout reproducibility test.
4. Finish the README, Devpost draft, technical explanation, and video script.
5. Verify public links, licences, run commands, checkpoints, and reports.

No new model feature is introduced on Day 4.

## 19. Acceptance criteria

The project is complete only when:

1. A clean checkout can execute the documented quick-start workflow.
2. Dataset and model licences are recorded.
3. Dataset audits and manifests are reproducible.
4. No checksum or source-group leakage is detected.
5. A baseline and at least one robust-training experiment are evaluated under identical splits.
6. The selected checkpoint maximizes the documented composite validation score.
7. Clean, per-condition, macro robust, pooled robust, worst-condition, and unseen-generator metrics are generated automatically.
8. False-positive and false-negative examples are documented.
9. The final model runs on the target laptop without CUDA.
10. PyTorch and published exported inference agree within tolerance.
11. The local web interface performs image upload, prediction, and transformation-stability comparison.
12. The public repository contains no private data, credentials, or unlicensed artifacts.
13. The README, Devpost draft, robustness table, error analysis, and video script are complete.
14. Measured trade-offs and known limitations are stated clearly.

## 20. Design principles

1. Complete and reproducible beats broad and unfinished.
2. Data alignment and evaluation integrity precede architectural complexity.
3. Every robustness claim requires condition-specific evidence.
4. Every optional module must earn its complexity through an ablation.
5. Clean and robust performance receive equal model-selection weight.
6. Unseen-generator evaluation is mandatory.
7. The web interface demonstrates the shared inference system rather than implementing a separate path.
8. All published results remain traceable to a configuration, manifest, split, checkpoint, and code revision.
