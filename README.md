# ProofLens

ProofLens is a research prototype for binary image classification between authentic and
AI-generated images. Its committed design evaluates whether predictions survive JPEG
compression, blur, resizing, noise, color jitter, and center cropping. It also keeps complete
generator families out of training for unseen-generator evaluation.

## Release status

The implementation, E0 through E4 experiments, validation-only selection and calibration,
untouched-test evaluation, parity-gated ONNX export, and Windows CPU acceptance are complete.
E2 was selected with a validation composite AUC of 0.9832. On the untouched test set it achieved
0.9788 clean ROC AUC and 0.9783 macro robust ROC AUC; see the
[`final robustness report`](docs/reports/final-robustness.md). Model weights and generated
artifacts are intentionally excluded from Git and distributed through the
[`prooflens-e2-rc1` GitHub Release](https://github.com/yangzhiloh/Tiktoky/releases/tag/prooflens-e2-rc1).
The miniature fixture workflow remains a software reproducibility check, not evidence of
primary-model performance.

## Requirements

- Python 3.11 or 3.12, as declared by `pyproject.toml`
- uv 0.12.0
- Git for the tracked-file release check
- Network access for SID-Set acquisition and the first DINOv2 download
- The pinned CC BY 4.0 AIGenImages2026 paired evaluation subset

CUDA is not required by the code or CI. CI exercises Python 3.11 on Windows and Linux with CPU
tests. OpenVINO is optional.

## Installation

### Windows PowerShell

```powershell
uv sync --locked --extra dev --python 3.11
```

### Linux shell

```bash
uv sync --locked --extra dev --python 3.11
```

`uv.lock` pins the complete direct and transitive dependency graph for Python 3.11 and 3.12,
including platform-specific distributions and SHA-256 hashes. `--locked` rejects any mismatch
between `pyproject.toml` and the committed lock instead of silently resolving new versions.

Run project commands through `uv run --locked --extra dev`, for example
`uv run --locked --extra dev python -m pytest -q`. This selects the synchronized environment
consistently on Windows and Linux without requiring shell activation.

## Fast offline reproduction

The task runner provides the same commands on Windows and Linux. `demo` is the one-click path: it
installs the locked environment, generates the fixture artifacts if they are absent, and launches
the local app. It downloads no dataset or pretrained weight.

```powershell
.\scripts\prooflens.ps1 demo
```

```bash
bash scripts/prooflens.sh demo
```

Both runners accept `setup`, `verify`, `artifacts`, and `demo`. PowerShell also accepts
`-PythonVersion` and `-Output`; the shell runner accepts those as its second and third positional
arguments. Generated demo files are placed under `artifacts/demo/`. Values produced by this
fixture run must not be reported as primary results.

## Dataset preparation

The full policy, licence status, expected directories, and acquisition boundaries are in
[`docs/datasets.md`](docs/datasets.md). Raw data stays below `data/raw/`, which Git ignores and
the release check rejects if tracked.

Before acquisition, run `.\scripts\prooflens.ps1 preflight` on Windows or
`bash scripts/prooflens.sh preflight` on Linux. The hardware, storage, credentials, licence,
dataset-layout, pilot, and recovery checklist is in
[`docs/task8-preflight.md`](docs/task8-preflight.md). This command does not download or train.

Acquire the pinned balanced SID-Set subset:

```text
python -m prooflens.cli acquire --config configs/data/sid_subset.yaml --output data/raw/sid_set
```

The smaller-run primary policy uses the pinned AIGenImages2026 paired validation subset at
`data/raw/aigenimages2026/val`. It supplies 559 authentic/synthetic pairs across 19 generator
families under CC BY 4.0. Pair members share a split group to prevent semantic leakage. SID-Set
remains an optional larger supplement rather than a requirement for this run.

Build, audit, and split the primary manifest:

```text
python -m prooflens.cli manifest --config configs/data/primary.yaml --output artifacts/manifests/primary.parquet
python -m prooflens.cli audit --manifest artifacts/manifests/primary.parquet --output artifacts/reports/data-audit
python -m prooflens.cli split --manifest artifacts/manifests/primary.parquet --output artifacts/manifests/primary-split.parquet --seed 17 --minimum-holdout-family-rows 20
```

The manifest policy requires both binary labels and at least three fake generator families from
approved generator-labelled sources. Splitting is grouped and checks exact and perceptual
duplicates before assigning train, validation, test, generator-validation, and generator-test
partitions.

## E0 through E4

Each configuration records the common split, seed, training schedule, model stage, transform
policy, and output directory.

```text
python -m prooflens.cli train --config configs/experiments/e0_frozen.yaml
python -m prooflens.cli train --config configs/experiments/e1_last2.yaml
python -m prooflens.cli train --config configs/experiments/e2_augmented.yaml
python -m prooflens.cli train --config configs/experiments/e3_consistency.yaml
python -m prooflens.cli train --config configs/experiments/e4_hard_mining.yaml
```

E0 trains the head, E1 unfreezes the final two DINOv2 blocks, E2 adds transformed
classification, E3 adds prediction and feature consistency, and E4 adds loss-guided hard
transformation selection.

Evaluate every candidate on the validation contract:

```text
python -m prooflens.cli evaluate --run artifacts/runs/e0 --suite clean-robust-generator --split validation
python -m prooflens.cli evaluate --run artifacts/runs/e1 --suite clean-robust-generator --split validation
python -m prooflens.cli evaluate --run artifacts/runs/e2 --suite clean-robust-generator --split validation
python -m prooflens.cli evaluate --run artifacts/runs/e3 --suite clean-robust-generator --split validation
python -m prooflens.cli evaluate --run artifacts/runs/e4 --suite clean-robust-generator --split validation
```

## Selection, calibration, and final reports

The project score is `0.50 * clean AUC + 0.50 * macro robust AUC`. Macro robust AUC first
averages severity AUCs within each of the six transform families, then gives each family equal
weight. Ties use worst-family AUC, then unseen-generator AUC, then lower model complexity.

```text
python -m prooflens.cli select --runs artifacts/runs/e0 artifacts/runs/e1 artifacts/runs/e2 artifacts/runs/e3 artifacts/runs/e4 --output artifacts/selection.json
python -m prooflens.cli calibrate --selection artifacts/selection.json --split validation --output artifacts/calibration.json
python -m prooflens.cli evaluate --selection artifacts/selection.json --suite clean-robust-generator --split test
python -m prooflens.cli evaluate-stress --selection artifacts/selection.json --split test --output artifacts/reports/stress
python -m prooflens.cli report --selection artifacts/selection.json --output artifacts/reports/final
```

Selection occurs before temperature scaling. Calibration uses validation predictions only. The
final report command consumes the selected run's test predictions and frozen calibration.

## ONNX export and local app

The publication gate always compares 32 validation or test images between PyTorch and ONNX. A
failed parity check does not publish the staged model.

```text
python -m prooflens.cli export --selection artifacts/selection.json --format onnx --verify 32 --output artifacts/export/prooflens.onnx
python -m prooflens.cli app --backend onnx --model artifacts/export/prooflens.onnx --calibration artifacts/calibration.json
```

For an optional OpenVINO smoke attempt, install the extra and repeat export with
`--format openvino`. CPU ONNX remains the required fallback.

```text
uv sync --locked --all-extras
uv run --locked --all-extras python -m prooflens.cli export --selection artifacts/selection.json --format openvino --verify 32 --output artifacts/export/prooflens.onnx
```

The app displays calibrated authentic and AI-generated probabilities, a threshold-relative
label, one selected transformation, and the absolute probability change. It is a demonstration,
not forensic proof.

The Windows CPU acceptance and clean-checkout evidence are recorded in the
[`Task 9 acceptance report`](docs/reports/acceptance-report.md). A visual browser automation
limitation did not affect the live Gradio API acceptance.

### Download the released model

The large model files remain outside Git history. With the GitHub CLI installed, download the
CPU demo artifacts from the repository root:

```powershell
New-Item -ItemType Directory -Path artifacts/export -Force | Out-Null
gh release download prooflens-e2-rc1 --repo yangzhiloh/Tiktoky --pattern prooflens.onnx --pattern artifact_manifest.json --pattern export_report.json --dir artifacts/export
gh release download prooflens-e2-rc1 --repo yangzhiloh/Tiktoky --pattern calibration.json --pattern selection.json --dir artifacts
```

Then launch the production-candidate demo:

```powershell
uv run --locked --extra dev python -m prooflens.cli app --backend onnx --model artifacts/export/prooflens.onnx --calibration artifacts/calibration.json
```

The release also provides `prooflens-e2-checkpoint.pt` for reproducibility. It is not required
for the ONNX demo. Verify downloaded files against the release's `SHA256SUMS.txt` before
redistribution.

## Artifact layout

```text
data/raw/sid_set/                         local SID-Set images and manifest
data/raw/aigenimages2026/val/             pinned paired multi-generator subset
artifacts/manifests/primary.parquet       canonical unsplit manifest
artifacts/manifests/primary-split.parquet grouped split used by every experiment
artifacts/runs/e0 ... e4/                 resolved config, metadata, checkpoints, predictions
artifacts/selection.json                  selected run and validation split provenance
artifacts/calibration.json                validation-fitted temperature and threshold
artifacts/reports/final/                  test metrics, table, and AUC plot
artifacts/reports/stress/                 supplemental redistribution stress results
artifacts/export/prooflens.onnx           parity-verified export, when available
artifacts/export/export_report.json       ONNX parity evidence, when available
```

`data/` and `artifacts/` are intentionally excluded from Git. Publish weights or large exports
only through separately reviewed release storage.

## Expected failures

The CLI normalizes failures by category:

- `configuration error:` for missing, malformed, or inconsistent arguments and artifacts
- `data integrity error:` for invalid datasets, manifests, splits, or prediction data
- `model error:` for training or export failures
- `prooflens error:` for other project-defined failures

Common corrective messages include:

- SID acquisition refuses to overwrite an existing output root. Remove or rename that complete
  local output only after deciding it is safe to do so.
- A missing or empty WildFake root reports both configured official acquisition sources and the
  expected placement.
- `--verify must be 32 for the publication parity gate` means the export sample count changed.
- `ONNX parity requires 32 validation or test images` means the selected split is too small.
- `selection has no calibration_path; run calibrate first` means export, report, or app setup is
  ahead of the required validation-only calibration step.
- The ONNX app requires `--model`; the Torch app requires `--checkpoint`; both require
  `--calibration`.

## Development and release checks

```text
uv run --locked --extra dev python -m ruff check src tests scripts
uv run --locked --extra dev python -m pytest -q
uv run --locked --extra dev python scripts/release_check.py --root .
```

The release scanner uses `git ls-files` inside a repository, so ignored and untracked local
datasets are not misreported as published files. It rejects tracked raw data, credentials,
private keys, model binaries, files over 100 MiB, incomplete notices, and incomplete required
documents.

## Licence and responsible use

ProofLens project code is MIT licensed. Third-party model and dataset terms remain separate and
are recorded in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Review those terms before
acquisition, use, or redistribution. See [`docs/model-card.md`](docs/model-card.md) for intended
use, evaluation status, limitations, and ethical considerations.
