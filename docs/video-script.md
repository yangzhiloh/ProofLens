# ProofLens demonstration video script

Target duration: 2 to 4 minutes. E0 through E4, selection, calibration, untouched-test reporting,
the parity-gated ONNX export, laptop and clean-checkout acceptance, and separately hosted model
artifacts are complete. The script is ready for a human to record and upload.

## 0:00 to 0:25, problem and scope

On screen: title, repository README, and one neutral authentic or generated example whose use is
permitted.

Narration:

"ProofLens is a research prototype that scores whether an image is authentic or AI-generated.
The goal is not just clean-image classification. We test whether the score survives ordinary
changes such as JPEG compression, blur, resizing, noise, color adjustment, and cropping. The
result is a research signal, not forensic proof."

## 0:25 to 0:55, data integrity

On screen: `docs/datasets.md`, the canonical manifest schema, and an audit or split summary that
contains no restricted image content.

Narration:

"Every image is represented by a canonical manifest row with dataset, label, generator,
source-group, licence, and hash metadata. Exact and perceptual duplicates stay in one partition.
Complete generator families are held out so transfer is measured separately from familiar
generators. Raw datasets never enter Git."

## 0:55 to 1:25, training contribution

On screen: E0 through E4 configuration names and the loss expression from the design.

Narration:

"The detector uses a DINOv2 base backbone with a small binary head. The experiment sequence
starts with a frozen backbone, then fine-tunes the final two blocks. Later stages add transformed
classification, prediction and feature consistency, and hard-transform selection. For each
selected training image, the hard stage examines three transformation candidates and trains on
the one that most reduces the correct margin while preserving family exploration."

## 1:25 to 1:55, measured evaluation

On screen: the final Task 8 untouched-test report, Task 7 selected-run provenance, and condition
table. Do not show fixture metrics as project results.

Narration:

"E2-on-SID was selected using validation data. On the untouched primary test set of 1,006 images,
clean ROC AUC was 0.9974 and family-macro robust ROC AUC was 0.9971. The worst tested condition
had 0.9966 AUC, while unseen-generator AUC was 0.9638. At the validation-selected threshold,
accuracy was 98.41 percent, with 14 false positives and 2 false negatives. These measurements
describe this frozen dataset and split, not every image domain or generator."

## 1:55 to 2:35, local app

On screen: launch the parity-verified ONNX app. Upload two or three permitted examples one at a
time, including an image that was already JPEG-compressed, blurred, cropped, or resized before
upload. Click Analyze picture for each example and show the resulting probabilities and
provenance.

Command shown:

```text
python -m prooflens.cli app --backend onnx --model artifacts/export/prooflens.onnx --calibration artifacts/calibration.json
```

Narration:

"The local app uses the same calibrated inference service as the command-line workflow. It
analyzes each image exactly as uploaded, even when the image has already been compressed,
blurred, cropped, recolored, or resized. It shows authentic and AI-generated probabilities, the
decision threshold, processing observations, and model provenance. The robustness claim comes
from the measured condition-level evaluation shown earlier, not from the interface applying a
second edit."

## 2:35 to 3:05, export and reproducibility

On screen: the ONNX parity report, CI matrix, and miniature command.

Narration:

"PyTorch is the reference implementation. The ONNX file is published only after a 32-sample
parity check. Windows and Linux CI run lint, the full tests, the offline miniature workflow, and
ONNX parity on Python 3.11. OpenVINO remains optional, while CPU ONNX is the required fallback."

## 3:05 to 3:30, limitations and close

On screen: model-card limitations and licence notices.

Narration:

"Performance is bounded by the audited datasets and generators. Domain shifts, screenshots,
compound edits, and unusual content can still change scores. False positives and false negatives
require human review. Dataset and model terms remain separate from the MIT project code, and
WildFake redistribution stays blocked until its licence is verified."

End on the verified repository URL: `https://github.com/yangzhiloh/ProofLens`. After uploading the
video, add its verified public URL to `docs/devpost-draft.md` before submitting.
