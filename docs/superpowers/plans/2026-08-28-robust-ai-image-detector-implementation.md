# Robust AI Image Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build ProofLens, a reproducible DINOv2-based detector that distinguishes authentic from AI-generated images and remains stable under the six required redistribution transformations.

**Architecture:** A canonical dataset manifest feeds leakage-safe grouped splits and paired clean/transformed batches. A DINOv2-B/14 backbone with a layer-normalization and linear head is trained first as a clean baseline, then with paired classification, representation consistency, and loss-guided hard-transformation selection. The same inference service powers evaluation, ONNX export, reporting, and a local Gradio application.

**Tech Stack:** Python 3.11, PyTorch, torchvision, Hugging Face Transformers and Datasets, Pillow, NumPy, pandas, scikit-learn, Pydantic, PyYAML, imagehash, matplotlib, seaborn, ONNX, ONNX Runtime CPU, optional OpenVINO 2026, Gradio, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-28-robust-ai-image-detector-design.md`

## Global Constraints

1. Keep the complete model below 2 billion parameters.
2. Use `facebook/dinov2-base` as the committed backbone, with 86.6 million parameters, 224-by-224 input, and a 768-dimensional class token.
3. Use `0` for authentic and `1` for AI-generated throughout the package.
4. Exclude SID-Set label `2` tampered images from primary binary training and evaluation.
5. Split source groups before generating transformations; no original image or derivative may cross partitions.
6. Implement JPEG qualities 90, 70, 50, 30; blur sigma 0.5, 1.0, 2.0; resize scales 0.5, 0.25; noise sigma 0.02, 0.05, 0.10; brightness, contrast, and saturation factors 0.8 to 1.2; center crop at 80 percent side length.
7. Use one transformation at a time for the primary robust score.
8. Select checkpoints with `0.5 * clean_auc + 0.5 * macro_robust_auc`.
9. Guarantee CPU inference on the Windows laptop; CUDA cannot be required.
10. Treat OpenVINO as optional acceleration. CPU ONNX Runtime is the required exported fallback.
11. Keep datasets, checkpoints, credentials, and generated reports out of git unless they are deliberately small fixtures.
12. Use test-first development, run the exact verification command in every task, and commit each independently testable deliverable.
13. Do not begin optional frequency or authentic-manifold branches until Tasks 1 through 15 pass and E4 results are available.
14. Implement custom project code under the MIT licence and record third-party licences separately.

---

## Four-Day Execution Map and Ownership

The reviewed Milestone 1 data pipeline is the common base for three implementation roles. Each role works only in its assigned branch and file boundaries. The integration owner coordinates shared files and milestone reviews. Human-only actions remain limited to dataset or service logins, accepting third-party licences, starting any approved cloud compute session, judging the resulting demo, recording the video, and publishing the final external assets.

| Day | Technical critical path | Human-only support | Exit evidence |
| --- | --- | --- | --- |
| 1 | Complete Tasks 1 through 7, freeze the manifest and split, and verify the model forward path | Download or mount approved datasets and verify licence terms | Data audit, zero-leakage split, transformation tests, DINOv2 smoke test |
| 2 | Complete Tasks 8 through 11 and run E0, E1, then E2 | Start approved cloud training only if laptop training is too slow | Three checkpoints, validation comparison, first robustness table and error gallery |
| 3 | Complete Tasks 12 through 15, run E3 and E4, repeat the leading configuration, calibrate, and export ONNX | Inspect false positives and false negatives for domain relevance | Local Gradio app, E0 through E4 comparison, final selection JSON, parity report |
| 4 | Complete Tasks 16 through 18, run laptop and clean-checkout acceptance, and freeze the repository | Record and upload the demo video, submit Devpost entry | Release check, acceptance report, public repository and submission assets |

If a daily exit condition slips, preserve the completed baseline and robustness table. Remove optional OpenVINO work first, then reduce repeated seeds to one additional seed. Do not remove grouped splitting, required transformations, ONNX CPU inference, or error analysis.

---

## Three-Role Delivery Workflow

Create all three role branches from the same reviewed Milestone 1 commit on `codex/prooflens-implementation`. A role may add files within its ownership boundary, but it must not edit another role's owned files. When a cross-role interface needs to change, record the request in the progress ledger and let the integration owner make or assign the shared edit after the current milestone review.

| Role | Branch | Tasks | Responsibilities | Exclusive implementation ownership |
| --- | --- | ---: | --- | --- |
| Model and training | `role/model-training` | 7 to 9 | DINOv2 detector, trainable stages, losses, hard-example mining, trainer, checkpoints, run metadata, and focused tests | `src/prooflens/models/`, `src/prooflens/training/`, `tests/unit/models/`, `tests/unit/training/` |
| Evaluation and inference | `role/evaluation-inference` | 10 to 13 | Prediction records, metrics, validation-only selection, calibration, robustness reports, inference backends, ONNX export, optional OpenVINO smoke path, and focused tests | `src/prooflens/evaluation/`, `src/prooflens/reporting/`, `src/prooflens/inference/` except the frozen `preprocess.py` contract, `src/prooflens/export/`, and matching unit and export-parity tests |
| Product and release | `role/product-release` | 14 to 18 | Gradio UI, CLI, experiment configurations, miniature workflow, documentation, licensing, CI, experiment execution records, laptop acceptance, demo script, and final release assets | `src/prooflens/web/`, `src/prooflens/cli.py`, `configs/experiments/`, `scripts/`, `.github/workflows/`, release documentation, web and small-workflow tests |

### Shared-file ownership

The integration owner exclusively edits the shared planning and foundation files: this implementation guide, the plan progress ledger, `pyproject.toml`, `.gitignore`, `src/prooflens/__init__.py`, `src/prooflens/config.py`, `src/prooflens/errors.py`, and the frozen Milestone 1 data pipeline. `README.md`, `LICENSE`, and `THIRD_PARTY_NOTICES.md` are assigned to the Product and release role once its branch begins. Other roles submit dependency or shared-interface requests through the ledger instead of editing these files concurrently.

Existing interfaces are merge contracts. Model and training consumes `PairedBatch` and shared preprocessing without changing them. Evaluation and inference consumes model outputs and checkpoints after the Model and training merge. Product and release consumes the evaluation and inference service after both earlier role merges. Any necessary exception requires an integration-owner ruling and a focused regression test before the shared file changes.

### Merge order and milestone gates

1. Merge `role/model-training` after the Tasks 7 to 9 milestone review and full gate.
2. Update `role/evaluation-inference` from the integration branch, then merge it after the Tasks 10 to 13 milestone review and full gate.
3. Update `role/product-release` from the integration branch, then merge it after the Tasks 14 to 18 milestone reviews and release gate.

Do not merge a later role first, and do not consolidate per task. Keep datasets, model weights, generated experiment artifacts, and credentials out of Git. Every role hands off exact commands, configuration changes, test results, and measured deviations at its milestone boundary.

---

## Locked File Map

### Project and configuration

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | Package metadata, dependency groups, pytest and Ruff settings |
| `.gitignore` | Exclude environments, data, artifacts, caches, and credentials |
| `src/prooflens/__init__.py` | Package version |
| `src/prooflens/config.py` | Typed YAML configuration and path resolution |
| `src/prooflens/errors.py` | Typed user, data, metric, training, and export failures |
| `src/prooflens/cli.py` | Cross-platform command entry point |
| `configs/data/sid_subset.yaml` | SID-Set acquisition and label policy |
| `configs/data/wildfake.yaml` | WildFake local adapter policy |
| `configs/data/cifake.yaml` | CIFAKE stress-test policy |
| `configs/data/primary.yaml` | Combined SID-Set and generator-labeled WildFake manifest policy |
| `configs/experiments/e0_frozen.yaml` | Frozen-backbone baseline |
| `configs/experiments/e1_last2.yaml` | Final-two-block fine-tuning |
| `configs/experiments/e2_augmented.yaml` | Random paired classification |
| `configs/experiments/e3_consistency.yaml` | Prediction and feature consistency |
| `configs/experiments/e4_hard_mining.yaml` | Three-candidate hard-transformation selection |

### Data

| Path | Responsibility |
| --- | --- |
| `src/prooflens/data/schema.py` | `ManifestRecord`, label constants, schema validation |
| `src/prooflens/data/licences.py` | Dataset licence registry and attribution data |
| `src/prooflens/data/adapters/base.py` | Dataset adapter protocol |
| `src/prooflens/data/adapters/sid_set.py` | SID-Set rows and binary label filtering |
| `src/prooflens/data/adapters/wildfake.py` | WildFake hierarchy and generator extraction |
| `src/prooflens/data/adapters/cifake.py` | CIFAKE folder labels and stress-test marker |
| `src/prooflens/data/acquire.py` | Streaming SID subset acquisition and local-root checks |
| `src/prooflens/data/manifest.py` | Adapter orchestration, image validation, and manifest persistence |
| `src/prooflens/data/audit.py` | Distribution and source-shortcut audit |
| `src/prooflens/data/hashing.py` | SHA-256 and perceptual hashes |
| `src/prooflens/data/splitting.py` | Duplicate grouping, generator holdout, grouped splits |
| `src/prooflens/data/transforms.py` | Canonical transform registry and execution |
| `src/prooflens/data/dataset.py` | Manifest-backed source image dataset |
| `src/prooflens/data/sampling.py` | Class, dataset, and generator-balanced sampling weights |
| `src/prooflens/data/collate.py` | Paired view sampling and tensor collation |

### Model and training

| Path | Responsibility |
| --- | --- |
| `src/prooflens/models/detector.py` | DINOv2 wrapper, class-token head, trainable stages |
| `src/prooflens/models/types.py` | `DetectorOutput`, `LossBreakdown`, `Prediction` |
| `src/prooflens/training/losses.py` | Clean, transformed, logit, and feature losses |
| `src/prooflens/training/hard_mining.py` | Candidate creation and margin-based selection |
| `src/prooflens/training/checkpoints.py` | Atomic save, load, and resume state |
| `src/prooflens/training/trainer.py` | Epoch loop, validation, scheduler, early stopping |
| `src/prooflens/training/run_metadata.py` | Dataset, split, config, environment, and git fingerprints |

### Evaluation and reporting

| Path | Responsibility |
| --- | --- |
| `src/prooflens/evaluation/predict.py` | Batched predictions with condition metadata |
| `src/prooflens/evaluation/metrics.py` | Clean, family, macro, pooled, and worst metrics |
| `src/prooflens/evaluation/calibration.py` | Temperature scaling fitted on validation logits |
| `src/prooflens/evaluation/select.py` | Composite checkpoint ranking and tie-breaking |
| `src/prooflens/reporting/tables.py` | CSV, JSON, and Markdown robustness tables |
| `src/prooflens/reporting/plots.py` | Condition and clean-versus-robust plots |
| `src/prooflens/reporting/gallery.py` | False-positive, false-negative, and instability galleries |

### Inference, export, and web

| Path | Responsibility |
| --- | --- |
| `src/prooflens/inference/preprocess.py` | Shared 224-pixel DINOv2 preprocessing |
| `src/prooflens/inference/service.py` | Backend-neutral prediction and stability analysis |
| `src/prooflens/inference/torch_backend.py` | PyTorch checkpoint inference |
| `src/prooflens/inference/onnx_backend.py` | ONNX Runtime CPU inference |
| `src/prooflens/export/onnx_export.py` | Dynamo ONNX export and parity verification |
| `src/prooflens/export/openvino_export.py` | Optional OpenVINO compile and smoke check |
| `src/prooflens/web/app.py` | Gradio Blocks application |

### Tests and release assets

| Path | Responsibility |
| --- | --- |
| `tests/fixtures/make_fixture_data.py` | Deterministic miniature real and fake images |
| `tests/unit/` | Focused contract tests for every module |
| `tests/integration/test_small_workflow.py` | Manifest-to-report miniature workflow |
| `tests/integration/test_export_parity.py` | PyTorch and ONNX numerical agreement |
| `tests/integration/test_web_service.py` | Upload and transformation-stability service path |
| `scripts/reproduce_small.py` | Clean-checkout miniature reproduction command |
| `scripts/release_check.py` | Licence, secrets, commands, and artifact checks |
| `.github/workflows/ci.yml` | CPU unit and integration tests on Windows and Ubuntu |
| `README.md` | Setup, data, training, evaluation, export, and demo instructions |
| `LICENSE` | MIT project licence |
| `THIRD_PARTY_NOTICES.md` | Model and dataset attributions |
| `docs/datasets.md` | Dataset versions, licences, labels, and acquisition steps |
| `docs/model-card.md` | Intended use, metrics, limitations, and ethics |
| `docs/devpost-draft.md` | Judging-aligned project description |
| `docs/video-script.md` | Two-to-four-minute demo script |

---

### Task 1: Package foundation and typed configuration

**Files:**

- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/prooflens/__init__.py`
- Create: `src/prooflens/config.py`
- Create: `src/prooflens/errors.py`
- Create: `configs/experiments/e0_frozen.yaml`
- Create: `tests/unit/test_config.py`

**Interfaces:**

- Consumes: The approved design specification.
- Produces: `load_config(path: Path) -> ExperimentConfig`, `ExperimentConfig.resolve(base: Path) -> ExperimentConfig`, installable `prooflens` package.

- [ ] **Step 1: Write the failing configuration tests**

```python
# tests/unit/test_config.py
from pathlib import Path

import pytest

from prooflens.config import ExperimentConfig, load_config


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  name: facebook/dinov2-base\nunknown: true\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_resolve_makes_manifest_and_output_absolute(tmp_path: Path) -> None:
    config = ExperimentConfig.model_validate({
        "seed": 17,
        "data": {"manifest": "artifacts/manifests/train.parquet"},
        "model": {"name": "facebook/dinov2-base", "stage": "head"},
        "training": {"epochs": 1, "batch_size": 2},
        "output_dir": "artifacts/runs/e0",
    })
    resolved = config.resolve(tmp_path)
    assert resolved.data.manifest == tmp_path / "artifacts/manifests/train.parquet"
    assert resolved.output_dir == tmp_path / "artifacts/runs/e0"
```

- [ ] **Step 2: Run the tests and confirm the missing package failure**

Run: `python -m pytest tests/unit/test_config.py -v`

Expected: FAIL during collection with `ModuleNotFoundError: No module named 'prooflens'`.

- [ ] **Step 3: Create package metadata and strict configuration models**

```toml
# pyproject.toml
[build-system]
requires = ["hatchling>=1.25"]
build-backend = "hatchling.build"

[project]
name = "prooflens"
version = "0.1.0"
requires-python = ">=3.11,<3.13"
dependencies = [
  "torch>=2.5,<3",
  "torchvision>=0.20,<1",
  "transformers>=4.46,<6",
  "datasets>=3,<5",
  "huggingface-hub>=0.27,<2",
  "pillow>=11,<13",
  "numpy>=2,<3",
  "pandas>=2.2,<3",
  "pyarrow>=18,<25",
  "scikit-learn>=1.5,<2",
  "pydantic>=2.9,<3",
  "pyyaml>=6,<7",
  "imagehash>=4.3,<5",
  "matplotlib>=3.9,<4",
  "seaborn>=0.13,<1",
  "safetensors>=0.5,<1",
  "gradio>=5,<7",
  "onnx>=1.17,<2",
  "onnxscript>=0.4,<1",
  "onnxruntime>=1.20,<2",
]

[project.optional-dependencies]
openvino = ["openvino>=2026.0,<2027"]
dev = ["pytest>=8,<9", "pytest-cov>=6,<8", "ruff>=0.9,<1"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"

[tool.ruff]
line-length = 100
target-version = "py311"
```

```python
# src/prooflens/config.py
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    manifest: Path


class ModelConfig(StrictModel):
    name: str = "facebook/dinov2-base"
    stage: Literal["head", "last2"] = "head"


class TrainingConfig(StrictModel):
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_gradient_norm: float = Field(default=1.0, gt=0)
    warmup_fraction: float = Field(default=0.10, ge=0, lt=1)
    early_stopping_patience: int = Field(default=2, ge=1)
    num_workers: int = Field(default=0, ge=0)


class TransformConfig(StrictModel):
    enabled: bool = False
    hard_mining: bool = False
    candidate_count: int = Field(default=3, ge=1, le=6)
    exploration_probability: float = Field(default=0.20, ge=0, le=1)


class LossConfig(StrictModel):
    clean_bce: float = Field(default=1.0, ge=0)
    transformed_bce: float = Field(default=0.0, ge=0)
    prediction_consistency: float = Field(default=0.0, ge=0)
    feature_consistency: float = Field(default=0.0, ge=0)


class ExperimentConfig(StrictModel):
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    transforms: TransformConfig = Field(default_factory=TransformConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    output_dir: Path

    def resolve(self, base: Path) -> "ExperimentConfig":
        raw = self.model_dump()
        raw["data"]["manifest"] = base / self.data.manifest
        raw["output_dir"] = base / self.output_dir
        return ExperimentConfig.model_validate(raw)


def load_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
```

Create a single typed error hierarchy so CLI exit-code mapping and tests use the same classes:

```python
# src/prooflens/errors.py
class ProofLensError(Exception):
    """Base class for expected project failures."""


class UserInputError(ProofLensError):
    """The user supplied an invalid argument or input image."""


class DataIntegrityError(ProofLensError):
    """Dataset contents violate a required invariant."""


class ManifestBuildError(DataIntegrityError):
    """A canonical manifest could not be built safely."""


class ImageDecodeError(DataIntegrityError):
    """An image cannot be decoded as RGB pixels."""


class LeakageError(DataIntegrityError):
    """Related source groups occur in multiple partitions."""


class MetricPartitionError(DataIntegrityError):
    """A requested metric partition is invalid."""


class TrainingError(ProofLensError):
    """Training or checkpoint recovery failed."""


class ExportError(ProofLensError):
    """Model export or numerical parity validation failed."""
```

Create `configs/experiments/e0_frozen.yaml` with these exact initial values:

```yaml
seed: 17
data:
  manifest: artifacts/manifests/primary-split.parquet
model:
  name: facebook/dinov2-base
  stage: head
training:
  epochs: 5
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.01
  gradient_accumulation_steps: 1
  max_gradient_norm: 1.0
  warmup_fraction: 0.10
  early_stopping_patience: 2
  num_workers: 0
transforms:
  enabled: false
  hard_mining: false
  candidate_count: 3
  exploration_probability: 0.20
loss:
  clean_bce: 1.0
  transformed_bce: 0.0
  prediction_consistency: 0.0
  feature_consistency: 0.0
output_dir: artifacts/runs/e0
```

- [ ] **Step 4: Install the editable package and run tests**

Run: `python -m pip install -e ".[dev]"`

Run: `python -m pytest tests/unit/test_config.py -v`

Expected: 2 tests PASS.

- [ ] **Step 5: Run lint and commit**

Run: `python -m ruff check src tests`

Expected: PASS with no diagnostics.

```bash
git add pyproject.toml .gitignore src/prooflens configs/experiments/e0_frozen.yaml tests/unit/test_config.py
git commit -m "chore: establish ProofLens package and configuration"
```

### Task 2: Canonical manifest schema and dataset adapters

**Files:**

- Create: `src/prooflens/data/schema.py`
- Create: `src/prooflens/data/licences.py`
- Create: `src/prooflens/data/adapters/base.py`
- Create: `src/prooflens/data/adapters/sid_set.py`
- Create: `src/prooflens/data/adapters/wildfake.py`
- Create: `src/prooflens/data/adapters/cifake.py`
- Create: `src/prooflens/data/manifest.py`
- Create: `tests/unit/data/test_schema.py`
- Create: `tests/unit/data/test_adapters.py`

**Interfaces:**

- Consumes: Local dataset roots or SID metadata mappings.
- Produces: `ManifestRecord`, `DatasetAdapter.scan() -> Iterator[ManifestRecord]`, `records_to_frame(records) -> pd.DataFrame`, `build_manifest(adapters, output_path, max_corrupt_fraction) -> ManifestBuildResult`.

- [ ] **Step 1: Write schema and adapter contract tests**

```python
# tests/unit/data/test_schema.py
from pathlib import Path

import pytest

from prooflens.data.schema import ManifestRecord, records_to_frame


def test_manifest_rejects_non_binary_primary_label() -> None:
    with pytest.raises(ValueError):
        ManifestRecord(
            sample_id="tampered-1", path=Path("image.jpg"), label=2,
            dataset_name="sid_set", dataset_version="main",
            generator_family="tampered", source_group_id="tampered-1",
            original_image_id="tampered-1", width=1024, height=1024,
            file_format="JPEG", licence_identifier="CC-BY-4.0",
        )


def test_records_to_frame_has_stable_column_order(valid_record: ManifestRecord) -> None:
    frame = records_to_frame([valid_record])
    assert frame.columns[:4].tolist() == ["sample_id", "path", "label", "dataset_name"]
```

```python
# tests/unit/data/test_adapters.py
def test_sid_adapter_excludes_tampered_label_two(tmp_path):
    rows = [
        {"img_id": "real-1", "label": 0, "image_path": tmp_path / "real.jpg"},
        {"img_id": "full_synthetic_1", "label": 1, "image_path": tmp_path / "fake.jpg"},
        {"img_id": "tampered_1", "label": 2, "image_path": tmp_path / "edit.jpg"},
    ]
    records = list(SidSetAdapter(version="main").scan_rows(rows))
    assert [record.label for record in records] == [0, 1]


def test_wildfake_adapter_reads_generator_from_hierarchy(wildfake_fixture):
    records = list(WildFakeAdapter(wildfake_fixture).scan())
    assert {record.generator_family for record in records if record.label == 1} == {"sdxl"}


def test_manifest_builder_stops_above_corrupt_limit(valid_adapter, corrupt_adapter, tmp_path):
    with pytest.raises(ManifestBuildError, match="corrupt fraction"):
        build_manifest([valid_adapter, corrupt_adapter], tmp_path / "manifest.parquet", max_corrupt_fraction=0.01)
```

- [ ] **Step 2: Verify failures for missing schema and adapters**

Run: `python -m pytest tests/unit/data/test_schema.py tests/unit/data/test_adapters.py -v`

Expected: FAIL with missing `prooflens.data` modules.

- [ ] **Step 3: Implement the strict manifest record and adapter protocol**

```python
# src/prooflens/data/schema.py
from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class ManifestRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sample_id: str
    path: Path
    label: Literal[0, 1]
    dataset_name: str
    dataset_version: str
    generator_family: str
    source_group_id: str
    original_image_id: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    file_format: str
    licence_identifier: str
    content_checksum: str = ""
    perceptual_hash: str = ""
    split: str = "unassigned"


def records_to_frame(records: list[ManifestRecord]) -> pd.DataFrame:
    return pd.DataFrame([record.model_dump(mode="json") for record in records])
```

```python
# src/prooflens/data/adapters/base.py
from collections.abc import Callable, Iterator
from typing import Protocol

from prooflens.data.schema import ManifestRecord


class DatasetAdapter(Protocol):
    scan: Callable[[], Iterator[ManifestRecord]]
```

Implement SID filtering as `0 -> authentic`, `1 -> generated`, and exclusion of `2`. Implement WildFake parsing through explicit real and fake directory mappings from configuration. Implement CIFAKE `REAL -> 0` and `FAKE -> 1`, with `generator_family="stable-diffusion-1.4"` for fake rows and `dataset_name="cifake_stress"` for every row.

Implement manifest orchestration with this contract:

```python
@dataclass(frozen=True)
class ManifestBuildResult:
    output_path: Path
    valid_count: int
    corrupt_count: int
    corrupt_paths: tuple[Path, ...]


def build_manifest(
    adapters: Sequence[DatasetAdapter],
    output_path: Path,
    max_corrupt_fraction: float = 0.01,
) -> ManifestBuildResult:
    """Validate image decoding, write Parquet atomically, and reject excess corruption."""
```

- [ ] **Step 4: Add fixture helpers and run adapter tests**

Run: `python -m pytest tests/unit/data/test_schema.py tests/unit/data/test_adapters.py -v`

Expected: All schema and adapter tests PASS.

- [ ] **Step 5: Commit the manifest contract**

```bash
git add src/prooflens/data tests/unit/data
git commit -m "feat: add canonical dataset manifest adapters"
```

### Task 3: Dataset acquisition and source-shortcut audit

**Files:**

- Create: `src/prooflens/data/acquire.py`
- Create: `src/prooflens/data/audit.py`
- Create: `configs/data/sid_subset.yaml`
- Create: `configs/data/wildfake.yaml`
- Create: `configs/data/cifake.yaml`
- Create: `configs/data/primary.yaml`
- Create: `tests/unit/data/test_acquire.py`
- Create: `tests/unit/data/test_audit.py`

**Interfaces:**

- Consumes: `saberzl/SID_Set` streaming rows or local dataset roots.
- Produces: `acquire_sid_subset(config, output_root) -> AcquisitionSummary`, `audit_manifest(frame) -> AuditReport`.

- [ ] **Step 1: Write acquisition selection and audit tests**

```python
def test_balanced_selector_stops_at_per_class_cap():
    rows = ({"img_id": f"{label}-{index}", "label": label} for index in range(10) for label in (0, 1, 2))
    selected = list(select_balanced_binary_rows(rows, per_class=3))
    assert sum(row["label"] == 0 for row in selected) == 3
    assert sum(row["label"] == 1 for row in selected) == 3
    assert all(row["label"] != 2 for row in selected)


def test_audit_flags_perfect_format_label_shortcut():
    frame = pd.DataFrame({
        "label": [0, 0, 1, 1],
        "file_format": ["PNG", "PNG", "JPEG", "JPEG"],
        "dataset_name": ["a", "a", "b", "b"],
        "width": [1024, 1024, 1024, 1024],
        "height": [768, 768, 768, 768],
    })
    report = audit_manifest(frame)
    assert "file_format" in report.perfect_shortcuts
    assert "dataset_name" in report.perfect_shortcuts
```

- [ ] **Step 2: Run tests and verify missing acquisition functions**

Run: `python -m pytest tests/unit/data/test_acquire.py tests/unit/data/test_audit.py -v`

Expected: FAIL with import errors.

- [ ] **Step 3: Implement streaming selection and explicit storage policy**

```python
@dataclass(frozen=True)
class AcquisitionSummary:
    output_root: Path
    manifest_path: Path
    counts: dict[int, int]
    dataset_revision: str
    licence_identifier: str
    config_sha256: str


def select_balanced_binary_rows(rows, per_class: int):
    counts = {0: 0, 1: 0}
    for row in rows:
        label = int(row["label"])
        if label not in counts or counts[label] >= per_class:
            continue
        counts[label] += 1
        yield row
        if all(count == per_class for count in counts.values()):
            return
```

`acquire_sid_subset` must call `load_dataset("saberzl/SID_Set", split="train", streaming=True)`, save a configurable default of 10,000 authentic and 10,000 fully synthetic images, preserve `img_id`, and write `acquisition.json` containing the Hugging Face revision, CC-BY-4.0 identifier, selected counts, and SHA-256 of the acquisition configuration.

WildFake acquisition must not pretend the dataset is automatically downloadable. It must validate a user-supplied ModelScope export root and direct the user to the official repository and ModelScope page when files are absent. CIFAKE remains a stress adapter and is not required for E0 through E4 training.

Create the combined primary policy with these exact gates:

```yaml
sources:
  - name: sid_set
    root: data/raw/sid_set
    allowed_labels: [0, 1]
  - name: wildfake
    root: data/raw/wildfake
    allowed_labels: [0, 1]
    minimum_generator_families: 3
maximum_corrupt_fraction: 0.01
require_both_labels: true
```

The primary manifest command must fail when WildFake or another generator-labeled approved source has not supplied at least three distinct fake generator families. One or more families remain available for training, one complete family supports generator-validation selection, and a different complete family remains untouched for final generator testing. This makes genuinely unseen-generator evaluation and test independence release requirements rather than optional metrics.

- [ ] **Step 4: Implement manifest audit statistics**

The audit must emit JSON and Markdown containing class counts, dataset counts, generator counts, width and height quantiles, file-format cross-tabs, missing metadata, exact duplicates, and categorical features that perfectly predict the label.

```python
@dataclass(frozen=True)
class AuditReport:
    row_count: int
    class_counts: dict[int, int]
    dataset_counts: dict[str, int]
    generator_counts: dict[str, int]
    dimension_quantiles: dict[str, dict[str, float]]
    missing_counts: dict[str, int]
    exact_duplicate_count: int
    perfect_shortcuts: tuple[str, ...]


def audit_manifest(frame: pd.DataFrame) -> AuditReport:
    categorical = ("dataset_name", "file_format", "generator_family")
    perfect = tuple(
        column for column in categorical
        if frame.groupby(column, dropna=False)["label"].nunique().max() == 1
    )
    quantiles = {
        column: frame[column].quantile([0.0, 0.25, 0.5, 0.75, 1.0]).to_dict()
        for column in ("width", "height")
    }
    return AuditReport(
        row_count=len(frame),
        class_counts=frame["label"].value_counts().sort_index().to_dict(),
        dataset_counts=frame["dataset_name"].value_counts().to_dict(),
        generator_counts=frame["generator_family"].value_counts().to_dict(),
        dimension_quantiles=quantiles,
        missing_counts=frame.isna().sum().to_dict(),
        exact_duplicate_count=int(frame["content_checksum"].duplicated().sum()),
        perfect_shortcuts=perfect,
    )


def write_audit(report: AuditReport, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "audit.json"
    markdown_path = output_dir / "audit.md"
    json_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    markdown_path.write_text(render_audit_markdown(report), encoding="utf-8")
    return json_path, markdown_path
```

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/unit/data/test_acquire.py tests/unit/data/test_audit.py -v`

Expected: PASS.

```bash
git add src/prooflens/data/acquire.py src/prooflens/data/audit.py configs/data tests/unit/data
git commit -m "feat: add reproducible data acquisition and audits"
```

### Task 4: Hashing, deduplication, and grouped splitting

**Files:**

- Create: `src/prooflens/data/hashing.py`
- Create: `src/prooflens/data/splitting.py`
- Create: `tests/unit/data/test_hashing.py`
- Create: `tests/unit/data/test_splitting.py`

**Interfaces:**

- Consumes: Canonical manifest DataFrame with valid local image paths.
- Produces: `enrich_hashes(frame) -> DataFrame`, `build_split_groups(frame, max_phash_distance) -> DataFrame`, `assign_grouped_splits(frame, policy) -> DataFrame`, `assert_no_leakage(frame) -> None`.

- [ ] **Step 1: Write exact and perceptual duplicate tests**

```python
def test_sha256_matches_identical_bytes(tmp_path):
    first = tmp_path / "a.bin"
    second = tmp_path / "b.bin"
    first.write_bytes(b"same")
    second.write_bytes(b"same")
    assert sha256_file(first) == sha256_file(second)


def test_cross_split_checksum_raises():
    frame = pd.DataFrame({
        "sample_id": ["a", "b"], "split": ["train", "test"],
        "content_checksum": ["same", "same"], "source_group_id": ["g1", "g2"],
    })
    with pytest.raises(LeakageError, match="content_checksum"):
        assert_no_leakage(frame)


def test_near_duplicate_chain_gets_one_split_group(near_duplicate_chain):
    grouped = build_split_groups(near_duplicate_chain, max_phash_distance=4)
    assert grouped.split_group_id.nunique() == 1
```

- [ ] **Step 2: Write generator-holdout and group-integrity tests**

```python
def test_holdout_generator_never_appears_in_train(sample_manifest):
    policy = SplitPolicy(
        seed=17,
        validation_fraction=0.15,
        test_fraction=0.15,
        generator_validation_families=frozenset({"sdxl"}),
        generator_test_families=frozenset({"flux"}),
    )
    split = assign_grouped_splits(sample_manifest, policy)
    assert set(split.loc[split.generator_family == "sdxl", "split"]) == {"generator_validation"}
    assert set(split.loc[split.generator_family == "flux", "split"]) == {"generator_test"}
    assert set(split.loc[split.split == "generator_validation", "label"]) == {0, 1}
    assert set(split.loc[split.split == "generator_test", "label"]) == {0, 1}
    assert split.groupby("source_group_id").split.nunique().max() == 1
```

- [ ] **Step 3: Implement hashing and grouped split assignment**

Use SHA-256 for exact identity and `imagehash.phash` for near-duplicate grouping. `build_split_groups` must use union-find over original `source_group_id`, identical SHA-256 values, and perceptual hashes within Hamming distance 4. Use a BK-tree for perceptual-hash neighbor lookup so the operation does not become quadratic. Assign each connected component a stable `split_group_id` derived from its lexicographically smallest sample ID before `GroupShuffleSplit`.

Create five independent partitions: `train`, `validation`, `test`, `generator_validation`, and `generator_test`. Complete fake generator families assigned to generator validation or generator test cannot occur elsewhere. Add disjoint real-image source groups to both generator partitions so ROC AUC is defined. Allocate 15 percent of remaining source groups to validation and 15 percent to test. `choose_holdout_generators` deterministically chooses the two largest eligible fake families by row count, with family name as a stable tie-break, assigning the largest to final generator test and the second largest to generator validation. Record both choices in split metadata. Fail if fewer than three fake families exist, if either holdout family has fewer than 100 samples, or if any evaluation partition lacks either binary label.

```python
@dataclass(frozen=True)
class SplitPolicy:
    seed: int
    validation_fraction: float
    test_fraction: float
    generator_validation_families: frozenset[str]
    generator_test_families: frozenset[str]
    generator_evaluation_real_fraction: float = 0.10


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash_file(path: Path) -> str:
    with Image.open(path) as image:
        return str(imagehash.phash(ImageOps.exif_transpose(image).convert("RGB")))


def enrich_hashes(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["content_checksum"] = result.path.map(lambda value: sha256_file(Path(value)))
    result["perceptual_hash"] = result.path.map(lambda value: perceptual_hash_file(Path(value)))
    return result


def assert_no_leakage(frame: pd.DataFrame) -> None:
    for column in ("content_checksum", "source_group_id", "split_group_id"):
        counts = frame.groupby(column)["split"].nunique()
        if (counts > 1).any():
            raise LeakageError(f"{column} appears in multiple splits")


def assign_grouped_splits(frame: pd.DataFrame, policy: SplitPolicy) -> pd.DataFrame:
    result = build_split_groups(frame, max_phash_distance=4)
    result["split"] = "train"
    fake_generator_validation = (
        (result.label == 1)
        & result.generator_family.isin(policy.generator_validation_families)
    )
    fake_generator_test = (
        (result.label == 1) & result.generator_family.isin(policy.generator_test_families)
    )
    validation_groups = set(result.loc[fake_generator_validation, "split_group_id"])
    test_groups = set(result.loc[fake_generator_test, "split_group_id"])
    if validation_groups & test_groups:
        raise LeakageError("a duplicate group links generator validation and generator test")
    result.loc[result.split_group_id.isin(validation_groups), "split"] = "generator_validation"
    result.loc[result.split_group_id.isin(test_groups), "split"] = "generator_test"

    real_groups = (
        result[(result.label == 0) & (result.split == "train")]
        .split_group_id.drop_duplicates().sort_values().to_numpy()
    )
    rng = np.random.default_rng(policy.seed)
    count = max(1, round(len(real_groups) * policy.generator_evaluation_real_fraction))
    chosen_real = rng.choice(real_groups, size=2 * count, replace=False)
    real_validation_groups = set(chosen_real[:count])
    real_test_groups = set(chosen_real[count:])
    result.loc[result.split_group_id.isin(real_validation_groups), "split"] = "generator_validation"
    result.loc[result.split_group_id.isin(real_test_groups), "split"] = "generator_test"

    remaining = result[result.split == "train"]
    train_validation, final_test = grouped_partition(
        remaining,
        fraction=policy.test_fraction,
        seed=policy.seed,
    )
    adjusted_validation = policy.validation_fraction / (1.0 - policy.test_fraction)
    final_train, final_validation = grouped_partition(
        train_validation,
        fraction=adjusted_validation,
        seed=policy.seed + 1,
    )
    result.loc[result.split_group_id.isin(set(final_validation.split_group_id)), "split"] = "validation"
    result.loc[result.split_group_id.isin(set(final_test.split_group_id)), "split"] = "test"
    assert_no_leakage(result)
    assert_every_evaluation_split_has_both_labels(result)
    return result
```

`grouped_partition` retries deterministic `GroupShuffleSplit` seeds up to 100 times until both output partitions contain both labels, then raises `DataIntegrityError` with class and group counts. The split writer stores policy, selected generator families, row counts, group counts, and manifest SHA-256 beside the Parquet file.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/unit/data/test_hashing.py tests/unit/data/test_splitting.py -v`

Expected: PASS.

```bash
git add src/prooflens/data/hashing.py src/prooflens/data/splitting.py tests/unit/data
git commit -m "feat: enforce duplicate-safe grouped splits"
```

### Task 5: Canonical transformation registry

**Files:**

- Create: `src/prooflens/data/transforms.py`
- Create: `tests/unit/data/test_transforms.py`

**Interfaces:**

- Consumes: `PIL.Image.Image`, `TransformSpec`, integer seed.
- Produces: `canonical_specs() -> tuple[TransformSpec, ...]`, `training_condition_probabilities() -> dict[str, float]`, `apply_transform(image, spec, seed) -> Image`, stable `condition_id` strings.

- [ ] **Step 1: Write registry completeness tests**

```python
def test_canonical_registry_has_all_required_conditions():
    ids = {spec.condition_id for spec in canonical_specs()}
    assert ids == {
        "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30",
        "blur_s0.5", "blur_s1.0", "blur_s2.0",
        "resize_x0.5", "resize_x0.25",
        "noise_s0.02", "noise_s0.05", "noise_s0.10",
        "color_jitter_20", "center_crop_80",
    }


def test_every_transform_preserves_original_dimensions(rgb_fixture):
    for spec in canonical_specs():
        transformed = apply_transform(rgb_fixture, spec, seed=17)
        assert transformed.size == rgb_fixture.size
        assert transformed.mode == "RGB"


def test_training_probabilities_weight_families_equally():
    probabilities = training_condition_probabilities()
    specs = {spec.condition_id: spec for spec in canonical_specs()}
    family_mass = {
        family: sum(
            probability for condition, probability in probabilities.items()
            if specs[condition].family == family
        )
        for family in {spec.family for spec in specs.values()}
    }
    assert family_mass == pytest.approx({family: 1 / 6 for family in family_mass})
```

- [ ] **Step 2: Write deterministic and severity tests**

```python
def test_noise_is_seed_deterministic(rgb_fixture):
    spec = get_spec("noise_s0.05")
    assert np.array_equal(
        np.asarray(apply_transform(rgb_fixture, spec, 9)),
        np.asarray(apply_transform(rgb_fixture, spec, 9)),
    )


def test_stronger_blur_reduces_edge_energy(rgb_fixture):
    mild = np.asarray(apply_transform(rgb_fixture, get_spec("blur_s0.5"), 1))
    strong = np.asarray(apply_transform(rgb_fixture, get_spec("blur_s2.0"), 1))
    assert edge_energy(strong) < edge_energy(mild)
```

- [ ] **Step 3: Implement typed specs and exact transformation functions**

```python
@dataclass(frozen=True)
class TransformSpec:
    family: Literal["jpeg", "blur", "resize", "noise", "color_jitter", "center_crop"]
    condition_id: str
    severity: float
    parameters: Mapping[str, float | int | str]
```

Use an in-memory Pillow JPEG round-trip with recorded `quality` and `subsampling=2`; torchvision Gaussian blur with an odd `2 * ceil(3 * sigma) + 1` kernel; Pillow bicubic resize; NumPy `default_rng(seed)` for `[0,1]` Gaussian noise; seed-driven brightness, contrast, and saturation factors; and an 80 percent side-length center crop followed by bicubic restoration.

Training samples one of the six families uniformly, then one severity uniformly within that family. Do not sample uniformly over all 14 condition IDs because that would overweight JPEG and noise relative to crop and color jitter, while the primary metric weights families equally.

- [ ] **Step 4: Run transform tests and commit**

Run: `python -m pytest tests/unit/data/test_transforms.py -v`

Expected: PASS for registry, dimensions, determinism, and severity ordering.

```bash
git add src/prooflens/data/transforms.py tests/unit/data/test_transforms.py
git commit -m "feat: implement canonical robustness transformations"
```

### Task 6: Manifest dataset and paired batch collation

**Files:**

- Create: `src/prooflens/data/dataset.py`
- Create: `src/prooflens/data/sampling.py`
- Create: `src/prooflens/data/collate.py`
- Create: `src/prooflens/inference/preprocess.py`
- Create: `tests/unit/data/test_dataset.py`
- Create: `tests/unit/data/test_sampling.py`
- Create: `tests/unit/data/test_collate.py`

**Interfaces:**

- Consumes: Assigned manifest rows, transformation sampler, DINOv2 image processor.
- Produces: `SourceImageDataset`, source-balanced sampling weights, `PairedBatch`, `PairedBatchCollator`, tensors shaped `[batch, 3, 224, 224]`.

- [ ] **Step 1: Write source loading and corrupt-image tests**

```python
def test_source_dataset_returns_image_label_and_manifest_metadata(manifest_fixture):
    item = SourceImageDataset(manifest_fixture)[0]
    assert item.image.mode == "RGB"
    assert item.label in (0, 1)
    assert item.sample_id == manifest_fixture.iloc[0].sample_id


def test_source_dataset_raises_typed_decode_error(corrupt_manifest):
    with pytest.raises(ImageDecodeError):
        SourceImageDataset(corrupt_manifest)[0]


def test_sampling_weights_balance_labels_and_fake_generators(imbalanced_manifest):
    weights = compute_sampling_weights(imbalanced_manifest)
    weighted = imbalanced_manifest.assign(weight=weights)
    assert weighted.groupby("label").weight.sum().to_dict() == pytest.approx({0: 0.5, 1: 0.5})
    fake = weighted[weighted.label == 1]
    generator_mass = fake.groupby(["dataset_name", "generator_family"]).weight.sum()
    assert generator_mass.max() == pytest.approx(generator_mass.min())
```

- [ ] **Step 2: Write paired collator tests with a network-free fake processor**

```python
def test_paired_collator_keeps_labels_and_shapes(source_items, fake_processor):
    collator = PairedBatchCollator(
        processor=fake_processor,
        sampler=FixedTransformSampler("jpeg_q50"),
        seed=17,
    )
    batch = collator(source_items)
    assert batch.clean_pixels.shape == (2, 3, 224, 224)
    assert batch.transformed_pixels.shape == (2, 3, 224, 224)
    assert torch.equal(batch.labels, torch.tensor([0.0, 1.0]))
    assert batch.condition_ids == ("jpeg_q50", "jpeg_q50")
```

- [ ] **Step 3: Implement shared preprocessing and paired collation**

Use `AutoImageProcessor.from_pretrained("facebook/dinov2-base")` in production. Keep the processor injectable so unit tests do not require network access. Derive per-item transformation seeds from the run seed, epoch, and stable sample ID hash.

Compute sampling strata as `(label, dataset_name)` for authentic images and `(label, dataset_name, generator_family)` for synthetic images. Give each label total sampling mass 0.5, divide that mass equally among its strata, and divide each stratum's mass equally among its rows. Feed these weights to a seeded `WeightedRandomSampler`. This prevents SID-Set or one prolific generator from dominating batches while retaining every approved training example.

```python
def compute_sampling_weights(frame: pd.DataFrame) -> np.ndarray:
    strata = frame.apply(
        lambda row: (
            f"real:{row.dataset_name}"
            if row.label == 0
            else f"fake:{row.dataset_name}:{row.generator_family}"
        ),
        axis=1,
    )
    weights = np.zeros(len(frame), dtype=np.float64)
    for label in (0, 1):
        label_mask = frame.label.to_numpy() == label
        label_strata = strata[label_mask]
        names = sorted(label_strata.unique())
        for name in names:
            mask = label_mask & (strata.to_numpy() == name)
            weights[mask] = 0.5 / (len(names) * int(mask.sum()))
    return weights
```

```python
@dataclass(frozen=True)
class SourceItem:
    image: Image.Image
    label: int
    sample_id: str
    generator_family: str


@dataclass(frozen=True)
class PairedBatch:
    clean_pixels: Tensor
    transformed_pixels: Tensor
    labels: Tensor
    sample_ids: tuple[str, ...]
    condition_ids: tuple[str, ...]


class PairedBatchCollator:
    def __init__(self, processor, sampler, seed: int) -> None:
        self.processor = processor
        self.sampler = sampler
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __call__(self, items: Sequence[SourceItem]) -> PairedBatch:
        specs = [self.sampler.sample(item.sample_id, self.epoch) for item in items]
        seeds = [stable_seed(self.seed, self.epoch, item.sample_id) for item in items]
        transformed = [
            apply_transform(item.image, spec, item_seed)
            for item, spec, item_seed in zip(items, specs, seeds, strict=True)
        ]
        clean_pixels = self.processor(
            images=[item.image for item in items], return_tensors="pt"
        )["pixel_values"]
        transformed_pixels = self.processor(images=transformed, return_tensors="pt")["pixel_values"]
        return PairedBatch(
            clean_pixels=clean_pixels,
            transformed_pixels=transformed_pixels,
            labels=torch.tensor([item.label for item in items], dtype=torch.float32),
            sample_ids=tuple(item.sample_id for item in items),
            condition_ids=tuple(spec.condition_id for spec in specs),
        )
```

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/unit/data/test_dataset.py tests/unit/data/test_sampling.py tests/unit/data/test_collate.py -v`

Expected: PASS.

```bash
git add src/prooflens/data/dataset.py src/prooflens/data/sampling.py src/prooflens/data/collate.py src/prooflens/inference/preprocess.py tests/unit/data
git commit -m "feat: add paired clean and transformed batches"
```

### Task 7: DINOv2 detector and trainable stages

**Files:**

- Create: `src/prooflens/models/types.py`
- Create: `src/prooflens/models/detector.py`
- Create: `tests/unit/models/test_detector.py`

**Interfaces:**

- Consumes: `[batch, 3, 224, 224]` float tensors.
- Produces: `DetectorOutput(logits: Tensor[batch], features: Tensor[batch, 768])`, `set_trainable_stage(stage)`.

- [ ] **Step 1: Write output and stage-freezing tests using a tiny DINO configuration**

```python
def test_detector_returns_binary_logits_and_normalized_features(tiny_dino):
    model = DinoDetector(backbone=tiny_dino, hidden_size=32)
    output = model(torch.randn(2, 3, 28, 28))
    assert output.logits.shape == (2,)
    assert output.features.shape == (2, 32)
    assert torch.allclose(output.features.norm(dim=1), torch.ones(2), atol=1e-5)


def test_last2_stage_unfreezes_only_final_two_blocks(tiny_dino):
    model = DinoDetector(backbone=tiny_dino, hidden_size=32)
    model.set_trainable_stage("last2")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert any("encoder.layer.2" in name for name in trainable)
    assert any("encoder.layer.3" in name for name in trainable)
    assert not any("encoder.layer.1" in name for name in trainable)
    assert any("classifier" in name for name in trainable)
```

- [ ] **Step 2: Run tests and confirm model module is missing**

Run: `python -m pytest tests/unit/models/test_detector.py -v`

Expected: FAIL with missing `prooflens.models.detector`.

- [ ] **Step 3: Implement the detector wrapper**

```python
@dataclass(frozen=True)
class DetectorOutput:
    logits: Tensor
    features: Tensor


@dataclass(frozen=True)
class LossBreakdown:
    total: Tensor
    clean_bce: Tensor
    transformed_bce: Tensor
    prediction_consistency: Tensor
    feature_consistency: Tensor


@dataclass(frozen=True)
class Prediction:
    probability_ai: float
    probability_real: float
    confidence: float
    logit: float
    model_version: str
    inference_ms: float


@dataclass(frozen=True)
class StabilityResult:
    condition_id: str
    clean: Prediction
    transformed: Prediction
    absolute_change: float


class DinoDetector(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int = 768) -> None:
        super().__init__()
        self.backbone = backbone
        self.feature_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, pixel_values: Tensor) -> DetectorOutput:
        hidden = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
        normalized = self.feature_norm(hidden)
        logits = self.classifier(normalized).squeeze(-1)
        return DetectorOutput(logits=logits, features=F.normalize(normalized, dim=1))
```

Add `from_pretrained()` using `Dinov2Model.from_pretrained("facebook/dinov2-base")`. `set_trainable_stage("head")` freezes the complete backbone. `set_trainable_stage("last2")` freezes the backbone and then unfreezes `backbone.encoder.layer[-2:]`.

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/unit/models/test_detector.py -v`

Expected: PASS.

```bash
git add src/prooflens/models tests/unit/models
git commit -m "feat: add staged DINOv2 detector"
```

### Task 8: Survival losses and hard-transformation selection

**Files:**

- Create: `src/prooflens/training/losses.py`
- Create: `src/prooflens/training/hard_mining.py`
- Create: `tests/unit/training/test_losses.py`
- Create: `tests/unit/training/test_hard_mining.py`

**Interfaces:**

- Consumes: Clean and transformed `DetectorOutput`, labels, three candidate condition families.
- Produces: `LossBreakdown`, `HardTransformSelection`, differentiable total loss.

- [ ] **Step 1: Write exact loss-component tests**

```python
def test_identical_views_have_zero_consistency_losses():
    output = DetectorOutput(
        logits=torch.tensor([0.2, -0.3]),
        features=F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1),
    )
    result = compute_survival_loss(output, output, torch.tensor([1.0, 0.0]))
    assert result.prediction_consistency.item() == pytest.approx(0.0)
    assert result.feature_consistency.item() == pytest.approx(0.0)


def test_total_matches_approved_weights():
    result = compute_survival_loss(clean_output, transformed_output, labels)
    expected = (
        result.clean_bce + result.transformed_bce
        + 0.25 * result.prediction_consistency
        + 0.10 * result.feature_consistency
    )
    assert torch.allclose(result.total, expected)
```

- [ ] **Step 2: Write correct-margin selection tests**

```python
def test_miner_selects_lowest_correct_margin():
    logits_by_condition = {
        "jpeg_q30": torch.tensor([2.0, -2.0]),
        "blur_s2.0": torch.tensor([-1.0, 1.0]),
        "noise_s0.10": torch.tensor([0.5, -0.5]),
    }
    labels = torch.tensor([1.0, 0.0])
    selection = select_lowest_margin(logits_by_condition, labels)
    assert selection == ("blur_s2.0", "blur_s2.0")
```

- [ ] **Step 3: Implement approved losses and margin definition**

Use `binary_cross_entropy_with_logits` for clean and transformed labels, mean-squared error for logits, and `1 - cosine_similarity` for features. Define correct margin as `logit` for label 1 and `-logit` for label 0. Select the smallest correct margin independently for every sample.

```python
@dataclass(frozen=True)
class SurvivalLossWeights:
    clean_bce: float = 1.0
    transformed_bce: float = 1.0
    prediction_consistency: float = 0.25
    feature_consistency: float = 0.10


def compute_survival_loss(
    clean: DetectorOutput,
    transformed: DetectorOutput,
    labels: Tensor,
    weights: SurvivalLossWeights = SurvivalLossWeights(),
) -> LossBreakdown:
    clean_bce = F.binary_cross_entropy_with_logits(clean.logits, labels)
    transformed_bce = F.binary_cross_entropy_with_logits(transformed.logits, labels)
    prediction_consistency = F.mse_loss(clean.logits, transformed.logits)
    feature_consistency = (1 - F.cosine_similarity(clean.features, transformed.features)).mean()
    total = (
        weights.clean_bce * clean_bce
        + weights.transformed_bce * transformed_bce
        + weights.prediction_consistency * prediction_consistency
        + weights.feature_consistency * feature_consistency
    )
    return LossBreakdown(
        total=total,
        clean_bce=clean_bce,
        transformed_bce=transformed_bce,
        prediction_consistency=prediction_consistency,
        feature_consistency=feature_consistency,
    )


def correct_margin(logits: Tensor, labels: Tensor) -> Tensor:
    return torch.where(labels == 1, logits, -logits)
```

- [ ] **Step 4: Implement three-family candidate sampling**

Sample three distinct families without replacement and one listed severity per family. Return selected condition IDs and transformed tensors with no gradient through the selection passes. The final selected forward pass participates in gradient calculation. Use the configured 0.20 exploration probability to choose a uniformly random candidate instead of the hardest one for that sample. This preserves uniform family exploration and reduces collapse onto the currently most destructive codec. Log both candidate-family and selected-family proportions every epoch; warn when one selected family exceeds 60 percent.

```python
@dataclass(frozen=True)
class HardTransformSelection:
    condition_ids: tuple[str, ...]
    transformed_images: tuple[Image.Image, ...]


class HardTransformMiner:
    def __init__(
        self,
        registry: Sequence[TransformSpec],
        seed: int,
        candidate_count: int = 3,
        exploration_probability: float = 0.20,
    ) -> None:
        self.by_family = group_specs_by_family(registry)
        self.seed = seed
        self.candidate_count = candidate_count
        self.exploration_probability = exploration_probability

    def sample_candidates(
        self, sample_ids: Sequence[str], epoch: int
    ) -> tuple[tuple[TransformSpec, ...], ...]:
        selections = []
        families = tuple(sorted(self.by_family))
        for sample_id in sample_ids:
            rng = random.Random(stable_seed(self.seed, epoch, sample_id))
            chosen_families = rng.sample(families, k=self.candidate_count)
            selections.append(tuple(rng.choice(self.by_family[name]) for name in chosen_families))
        return tuple(selections)

    def select(
        self,
        candidate_logits: Tensor,
        candidate_condition_ids: Sequence[Sequence[str]],
        labels: Tensor,
        sample_ids: Sequence[str],
        epoch: int,
    ) -> tuple[str, ...]:
        if candidate_logits.shape != (len(labels), self.candidate_count):
            raise ValueError("candidate logits must have shape [batch, candidate_count]")
        margins = torch.where(labels[:, None] == 1, candidate_logits, -candidate_logits)
        selected_indices = margins.argmin(dim=1).tolist()
        for index, sample_id in enumerate(sample_ids):
            rng = random.Random(stable_seed(self.seed, epoch, sample_id, "explore"))
            if rng.random() < self.exploration_probability:
                selected_indices[index] = rng.randrange(self.candidate_count)
        return tuple(
            candidate_condition_ids[row][candidate_index]
            for row, candidate_index in enumerate(selected_indices)
        )
```

The trainer batches the `batch_size * candidate_count` candidate images into one no-gradient forward pass, reshapes logits to `[batch_size, candidate_count]`, calls `select`, then rebuilds only the chosen transformed batch for the gradient-bearing forward pass.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/unit/training/test_losses.py tests/unit/training/test_hard_mining.py -v`

Expected: PASS.

```bash
git add src/prooflens/training/losses.py src/prooflens/training/hard_mining.py tests/unit/training
git commit -m "feat: add survival loss and hard transform mining"
```

### Task 9: Resumable trainer and experiment metadata

**Files:**

- Create: `src/prooflens/training/checkpoints.py`
- Create: `src/prooflens/training/run_metadata.py`
- Create: `src/prooflens/training/trainer.py`
- Create: `tests/unit/training/test_checkpoints.py`
- Create: `tests/integration/test_tiny_training.py`

**Interfaces:**

- Consumes: `ExperimentConfig`, model, paired loaders, optimizer, metric callback.
- Produces: atomic checkpoints, `history.jsonl`, `run_metadata.json`, best checkpoint path.

- [ ] **Step 1: Write atomic checkpoint round-trip test**

```python
def test_checkpoint_restores_model_optimizer_and_epoch(tmp_path, tiny_model):
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    manager = CheckpointManager(tmp_path)
    path = manager.save("epoch-1", tiny_model, optimizer, epoch=1, global_step=7, config_hash="abc")
    restored = manager.load(path, tiny_model, optimizer)
    assert restored.epoch == 1
    assert restored.global_step == 7
    assert restored.config_hash == "abc"
    assert not path.with_suffix(".tmp").exists()
```

- [ ] **Step 2: Write one-epoch integration test**

```python
def test_tiny_training_emits_checkpoint_and_history(tiny_training_case):
    result = run_training(tiny_training_case.config)
    assert result.best_checkpoint.exists()
    assert (result.output_dir / "history.jsonl").exists()
    history = [json.loads(line) for line in (result.output_dir / "history.jsonl").read_text().splitlines()]
    assert history[-1]["epoch"] == 1
    assert math.isfinite(history[-1]["train_loss"])
```

- [ ] **Step 3: Implement complete run metadata**

Record the git commit, Python and package versions, operating system, device, dataset manifest SHA-256, split SHA-256, configuration SHA-256, seed, start time, and backbone identifier. Write metadata before the first optimizer step.

```python
@dataclass(frozen=True)
class RunMetadata:
    git_commit: str
    python_version: str
    package_versions: dict[str, str]
    operating_system: str
    device: str
    manifest_sha256: str
    split_sha256: str
    config_sha256: str
    seed: int
    started_at_utc: str
    backbone: str


def collect_run_metadata(config: ExperimentConfig) -> RunMetadata:
    packages = ("torch", "torchvision", "transformers", "datasets", "onnxruntime")
    return RunMetadata(
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        python_version=platform.python_version(),
        package_versions={name: metadata.version(name) for name in packages},
        operating_system=platform.platform(),
        device=detect_device_name(),
        manifest_sha256=sha256_file(config.data.manifest),
        split_sha256=sha256_file(config.data.manifest),
        config_sha256=sha256_text(config.model_dump_json()),
        seed=config.seed,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        backbone=config.model.name,
    )
```

- [ ] **Step 4: Implement trainer phases and safe device behavior**

Use AdamW, configurable learning rate, gradient accumulation, gradient clipping, 10 percent linear warmup followed by cosine decay, and validation after each epoch. Stop after the configured number of epochs without validation composite improvement. Enable automatic mixed precision only for supported CUDA devices. On CPU or Intel paths, use float32 and default to zero worker subprocesses for Windows reliability. When a CUDA out-of-memory exception occurs, terminate with a structured message containing the current batch size and recommended smaller value; do not silently alter scientific configuration inside a run.

```python
@dataclass(frozen=True)
class TrainingResult:
    output_dir: Path
    best_checkpoint: Path
    best_composite_score: float


@dataclass(frozen=True)
class ValidationSnapshot:
    clean_auc: float
    macro_robust_auc: float
    composite_score: float
    metrics: Mapping[str, float]


def run_training(config: ExperimentConfig) -> TrainingResult:
    components = build_training_components(config)
    return Trainer(config=config, **components).fit()


class Trainer:
    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.collator.set_epoch(epoch)
        running_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(self.train_loader, start=1):
            raw_loss = self.compute_batch_loss(batch)
            scaled_loss = raw_loss / self.config.training.gradient_accumulation_steps
            scaled_loss.backward()
            running_loss += float(raw_loss.detach())
            should_step = (
                step % self.config.training.gradient_accumulation_steps == 0
                or step == len(self.train_loader)
            )
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.max_gradient_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
        return running_loss / max(1, len(self.train_loader))

    def validate(self, epoch: int) -> ValidationSnapshot:
        return self.validation_callback(self.model, epoch)

    def fit(self) -> TrainingResult:
        best_score = -math.inf
        best_checkpoint = None
        epochs_without_improvement = 0
        for epoch in range(self.start_epoch, self.config.training.epochs + 1):
            train_loss = self.train_epoch(epoch)
            validation = self.validate(epoch)
            checkpoint = self.checkpoints.save_epoch(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                global_step=self.global_step,
                config=self.config,
            )
            append_history(self.output_dir, epoch, train_loss, validation)
            if validation.composite_score > best_score:
                best_score = validation.composite_score
                best_checkpoint = self.checkpoints.mark_best(checkpoint)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.training.early_stopping_patience:
                    break
        if best_checkpoint is None:
            raise TrainingError("training produced no checkpoint")
        return TrainingResult(self.output_dir, best_checkpoint, best_score)
```

- [ ] **Step 5: Run training tests and commit**

Run: `python -m pytest tests/unit/training/test_checkpoints.py tests/integration/test_tiny_training.py -v`

Expected: PASS.

```bash
git add src/prooflens/training tests/unit/training tests/integration/test_tiny_training.py
git commit -m "feat: add resumable experiment trainer"
```

### Task 10: Robust prediction records and metric aggregation

**Files:**

- Create: `src/prooflens/evaluation/predict.py`
- Create: `src/prooflens/evaluation/metrics.py`
- Create: `src/prooflens/evaluation/select.py`
- Create: `tests/unit/evaluation/test_metrics.py`
- Create: `tests/unit/evaluation/test_select.py`

**Interfaces:**

- Consumes: Sample ID, label, logit, generator, family, condition ID, checkpoint ID.
- Produces: prediction Parquet, `MetricReport`, ranked checkpoints.

- [ ] **Step 1: Write macro, pooled, and worst-condition tests**

```python
def test_macro_robust_weights_families_equally(prediction_frame):
    report = compute_metrics(prediction_frame)
    expected = np.mean([
        report.family_auc["jpeg"], report.family_auc["blur"],
        report.family_auc["resize"], report.family_auc["noise"],
        report.family_auc["color_jitter"], report.family_auc["center_crop"],
    ])
    assert report.macro_robust_auc == pytest.approx(expected)


def test_metric_rejects_single_class_condition():
    with pytest.raises(MetricPartitionError, match="both labels"):
        compute_condition_auc(pd.DataFrame({"label": [1, 1], "score": [0.2, 0.8]}))
```

- [ ] **Step 2: Write checkpoint tie-break tests**

```python
def test_checkpoint_selection_uses_worst_family_then_unseen_auc():
    candidates = [
        Candidate("a", clean_auc=0.9, macro_robust_auc=0.8, worst_family_auc=0.6, unseen_auc=0.8),
        Candidate("b", clean_auc=0.9, macro_robust_auc=0.8, worst_family_auc=0.7, unseen_auc=0.7),
    ]
    assert select_best(candidates).checkpoint_id == "b"
```

- [ ] **Step 3: Implement stable prediction schema and metrics**

Store raw logits and sigmoid scores. During experiment selection, compute clean and transformed metrics only from `split="validation"` and unseen-generator AUC only from `split="generator_validation"`. After the model configuration and calibration are frozen, run the same function with `evaluation_split="test"` and `generator_split="generator_test"` exactly once for the final report. Compute clean AUC from `condition_id="clean"`; severity AUC by condition; family AUC as the mean of that family's condition AUC values; macro robust as the mean of six family values; pooled robust from every non-clean row; and worst condition and family as minima.

```python
@dataclass(frozen=True)
class PredictionRecord:
    sample_id: str
    label: int
    logit: float
    score: float
    split: str
    generator_family: str
    transform_family: str
    condition_id: str
    checkpoint_id: str


@dataclass(frozen=True)
class MetricReport:
    clean_auc: float
    condition_auc: dict[str, float]
    family_auc: dict[str, float]
    macro_robust_auc: float
    pooled_robust_auc: float
    worst_condition_auc: float
    worst_family_auc: float
    unseen_generator_auc: float
    composite_score: float
    model_parameters: int = 0
    inference_ms_median: float = float("nan")


@dataclass(frozen=True)
class Candidate:
    checkpoint_id: str
    clean_auc: float
    macro_robust_auc: float
    worst_family_auc: float
    unseen_auc: float
    parameter_count: int = 0

    @property
    def composite_score(self) -> float:
        return 0.5 * self.clean_auc + 0.5 * self.macro_robust_auc


def select_best(candidates: Sequence[Candidate]) -> Candidate:
    if not candidates:
        raise ValueError("at least one checkpoint candidate is required")
    return max(
        candidates,
        key=lambda item: (
            item.composite_score,
            item.worst_family_auc,
            item.unseen_auc,
            -item.parameter_count,
            item.checkpoint_id,
        ),
    )


def _auc(frame: pd.DataFrame) -> float:
    if frame.label.nunique() != 2:
        raise MetricPartitionError("metric partition must contain both labels")
    return float(roc_auc_score(frame.label, frame.score))


def compute_metrics(
    predictions: pd.DataFrame,
    evaluation_split: str = "validation",
    generator_split: str = "generator_validation",
) -> MetricReport:
    evaluation = predictions[predictions.split == evaluation_split]
    clean_auc = _auc(evaluation[evaluation.condition_id == "clean"])
    robust = evaluation[evaluation.condition_id != "clean"]
    condition_auc = {
        name: _auc(group) for name, group in robust.groupby("condition_id", sort=True)
    }
    condition_family = robust[["condition_id", "transform_family"]].drop_duplicates()
    family_auc = {
        family: float(np.mean([
            condition_auc[name]
            for name in condition_family.loc[
                condition_family.transform_family == family, "condition_id"
            ]
        ]))
        for family in sorted(condition_family.transform_family.unique())
    }
    macro_robust_auc = float(np.mean(list(family_auc.values())))
    unseen = predictions[
        (predictions.split == generator_split) & (predictions.condition_id == "clean")
    ]
    return MetricReport(
        clean_auc=clean_auc,
        condition_auc=condition_auc,
        family_auc=family_auc,
        macro_robust_auc=macro_robust_auc,
        pooled_robust_auc=_auc(robust),
        worst_condition_auc=min(condition_auc.values()),
        worst_family_auc=min(family_auc.values()),
        unseen_generator_auc=_auc(unseen),
        composite_score=0.5 * clean_auc + 0.5 * macro_robust_auc,
    )
```

- [ ] **Step 4: Run tests and commit**

Run: `python -m pytest tests/unit/evaluation/test_metrics.py tests/unit/evaluation/test_select.py -v`

Expected: PASS.

```bash
git add src/prooflens/evaluation tests/unit/evaluation
git commit -m "feat: add robust evaluation and checkpoint selection"
```

### Task 11: Calibration, robustness tables, plots, and error galleries

**Files:**

- Create: `src/prooflens/evaluation/calibration.py`
- Create: `src/prooflens/reporting/tables.py`
- Create: `src/prooflens/reporting/plots.py`
- Create: `src/prooflens/reporting/gallery.py`
- Create: `tests/unit/evaluation/test_calibration.py`
- Create: `tests/unit/reporting/test_reporting.py`

**Interfaces:**

- Consumes: Validation logits, prediction records, manifest paths, `MetricReport`.
- Produces: `calibration.json`, `metrics.json`, `robustness.csv`, `robustness.md`, PNG plots, HTML error gallery.

- [ ] **Step 1: Write temperature-scaling test**

```python
def test_temperature_scaling_does_not_increase_validation_nll():
    logits = torch.tensor([4.0, 3.0, -4.0, -3.0])
    labels = torch.tensor([1.0, 0.0, 0.0, 1.0])
    scaler = fit_temperature(logits, labels)
    before = F.binary_cross_entropy_with_logits(logits, labels)
    after = F.binary_cross_entropy_with_logits(scaler(logits), labels)
    assert after <= before + 1e-6
    assert scaler.temperature.item() > 0


def test_operating_threshold_is_fitted_on_validation_scores():
    scores = np.array([0.05, 0.20, 0.70, 0.95])
    labels = np.array([0, 0, 1, 1])
    threshold = select_operating_threshold(scores, labels)
    report = compute_threshold_metrics(scores, labels, threshold)
    assert 0.20 < threshold <= 0.70
    assert report.false_positives == 0
    assert report.false_negatives == 0
```

- [ ] **Step 2: Write table and gallery tests**

```python
def test_markdown_table_contains_required_rows(metric_report, tmp_path):
    path = write_robustness_markdown(metric_report, tmp_path / "robustness.md")
    text = path.read_text(encoding="utf-8")
    for name in ("Clean", "JPEG", "Blur", "Resize", "Noise", "Color jitter", "Center crop"):
        assert name in text


def test_gallery_selects_highest_confidence_errors(predictions):
    selected = select_error_cases(predictions, per_category=2)
    assert len(selected.false_positives) == 2
    assert selected.false_positives.score.is_monotonic_decreasing


def test_auc_plot_is_written(metric_report, tmp_path):
    path = write_auc_plot(metric_report, tmp_path / "auc.png")
    assert path.is_file()
    assert path.stat().st_size > 0
```

- [ ] **Step 3: Implement positive temperature parameterization and report writers**

Optimize `log_temperature` with LBFGS and expose `temperature = exp(log_temperature)`. After temperature fitting, choose a decision threshold from calibrated clean validation scores by maximizing Youden's J statistic, with the threshold closest to 0.5 as the deterministic tie-break. Store temperature, threshold, validation split hash, and fitting timestamp in `calibration.json`. Never refit either value on test data. Reports must include clean, every condition, six family aggregates, macro robust, pooled robust, worst family, worst condition, generator holdout, accuracy, precision, recall, F1, false positives, false negatives, model size, and inference time.

```python
class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp()

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature


def fit_temperature(logits: Tensor, labels: Tensor) -> TemperatureScaler:
    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS([scaler.log_temperature], lr=0.1, max_iter=100)

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(scaler(logits.detach()), labels.detach())
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler.eval()


@dataclass(frozen=True)
class ThresholdReport:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positives: int
    false_negatives: int


def select_operating_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    objective = true_positive_rate - false_positive_rate
    best = np.flatnonzero(objective == objective.max())
    return float(thresholds[min(best, key=lambda index: abs(thresholds[index] - 0.5))])


def compute_threshold_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> ThresholdReport:
    predicted = (scores >= threshold).astype(np.int64)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predicted, labels=[0, 1]
    ).ravel()
    return ThresholdReport(
        threshold=threshold,
        accuracy=float(accuracy_score(labels, predicted)),
        precision=float(precision_score(labels, predicted, zero_division=0)),
        recall=float(recall_score(labels, predicted, zero_division=0)),
        f1=float(f1_score(labels, predicted, zero_division=0)),
        false_positives=int(false_positive),
        false_negatives=int(false_negative),
    )


def metric_rows(report: MetricReport) -> list[tuple[str, float]]:
    rows = [("Clean", report.clean_auc)]
    display = {
        "jpeg": "JPEG", "blur": "Blur", "resize": "Resize", "noise": "Noise",
        "color_jitter": "Color jitter", "center_crop": "Center crop",
    }
    rows.extend((display[name], report.family_auc[name]) for name in display)
    rows.extend((f"Condition: {name}", value) for name, value in report.condition_auc.items())
    rows.extend([
        ("Macro robust", report.macro_robust_auc),
        ("Pooled robust", report.pooled_robust_auc),
        ("Worst family", report.worst_family_auc),
        ("Worst condition", report.worst_condition_auc),
        ("Unseen generator", report.unseen_generator_auc),
        ("Composite", report.composite_score),
    ])
    return rows


def write_robustness_markdown(
    report: MetricReport,
    path: Path,
    threshold_report: ThresholdReport | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| Partition | ROC AUC |", "| --- | ---: |"]
    lines.extend(f"| {name} | {value:.6f} |" for name, value in metric_rows(report))
    lines.extend([
        "", f"Model parameters: {report.model_parameters}",
        f"Median CPU inference time: {report.inference_ms_median:.3f} ms",
    ])
    if threshold_report is not None:
        lines.extend([
            f"Operating threshold: {threshold_report.threshold:.6f}",
            f"Accuracy: {threshold_report.accuracy:.6f}",
            f"Precision: {threshold_report.precision:.6f}",
            f"Recall: {threshold_report.recall:.6f}",
            f"F1: {threshold_report.f1:.6f}",
            f"False positives: {threshold_report.false_positives}",
            f"False negatives: {threshold_report.false_negatives}",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_metric_artifacts(
    report: MetricReport,
    threshold_report: ThresholdReport,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "robustness.csv"
    markdown_path = output_dir / "robustness.md"
    payload = {"ranking": asdict(report), "operating_point": asdict(threshold_report)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(metric_rows(report), columns=["partition", "roc_auc"]).to_csv(
        csv_path, index=False
    )
    write_robustness_markdown(report, markdown_path, threshold_report)
    return json_path, csv_path, markdown_path


def write_auc_plot(report: MetricReport, path: Path) -> Path:
    labels = ["Clean", *report.family_auc.keys()]
    values = [report.clean_auc, *report.family_auc.values()]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(labels, values)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("ROC AUC")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
```

- [ ] **Step 4: Generate safe image galleries**

Create thumbnails in the report directory rather than copying full datasets. Escape file paths and captions in generated HTML. Include sample ID, source dataset, generator, label, score, condition, and clean-to-transformed score change.

```python
@dataclass(frozen=True)
class ErrorCases:
    false_positives: pd.DataFrame
    false_negatives: pd.DataFrame
    unstable: pd.DataFrame


def select_error_cases(predictions: pd.DataFrame, per_category: int = 20) -> ErrorCases:
    false_positives = predictions[predictions.label == 0].nlargest(per_category, "score")
    false_negatives = predictions[predictions.label == 1].nsmallest(per_category, "score")
    clean = predictions[predictions.condition_id == "clean"][["sample_id", "score"]].rename(
        columns={"score": "clean_score"}
    )
    changed = predictions[predictions.condition_id != "clean"].merge(clean, on="sample_id")
    changed = changed.assign(absolute_change=(changed.score - changed.clean_score).abs())
    unstable = changed.nlargest(per_category, "absolute_change")
    return ErrorCases(false_positives, false_negatives, unstable)


def write_error_gallery(cases: ErrorCases, manifest: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_dir = output_dir / "thumbnails"
    thumbnail_dir.mkdir(exist_ok=True)
    metadata = manifest.set_index("sample_id")
    sections: list[str] = []
    for title, frame in (
        ("False positives", cases.false_positives),
        ("False negatives", cases.false_negatives),
        ("Most unstable", cases.unstable),
    ):
        cards = []
        for row in frame.itertuples():
            source_record = metadata.loc[row.sample_id]
            source = Path(source_record.path)
            thumb_name = hashlib.sha256(str(row.sample_id).encode()).hexdigest()[:16] + ".jpg"
            thumb_path = thumbnail_dir / thumb_name
            with Image.open(source) as image:
                image.convert("RGB").copy().resize((224, 224)).save(thumb_path, quality=85)
            caption = html.escape(
                f"{row.sample_id} | dataset={source_record.dataset_name} | path={source} | "
                f"label={row.label} | score={row.score:.4f} | condition={row.condition_id} | "
                f"generator={row.generator_family} | "
                f"absolute_change={getattr(row, 'absolute_change', float('nan')):.4f}"
            )
            cards.append(
                f'<figure><img src="thumbnails/{thumb_name}" alt="sample"><figcaption>{caption}</figcaption></figure>'
            )
        sections.append(f"<h2>{html.escape(title)}</h2>" + "".join(cards))
    gallery_path = output_dir / "error-gallery.html"
    gallery_path.write_text(
        "<!doctype html><meta charset='utf-8'><title>ProofLens errors</title>" + "".join(sections),
        encoding="utf-8",
    )
    return gallery_path
```

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/unit/evaluation/test_calibration.py tests/unit/reporting/test_reporting.py -v`

Expected: PASS.

```bash
git add src/prooflens/evaluation/calibration.py src/prooflens/reporting tests/unit/evaluation tests/unit/reporting
git commit -m "feat: add calibration and robustness reports"
```

### Task 12: Backend-neutral inference service

**Files:**

- Create: `src/prooflens/inference/service.py`
- Create: `src/prooflens/inference/torch_backend.py`
- Create: `tests/unit/inference/test_service.py`
- Create: `tests/unit/inference/test_torch_backend.py`

**Interfaces:**

- Consumes: PIL image, optional `TransformSpec`, checkpoint and calibration paths.
- Produces: `Prediction(probability_ai, probability_real, confidence, logit, model_version, inference_ms)` and `StabilityResult`.

- [ ] **Step 1: Write probability and stability tests**

```python
def test_prediction_probabilities_sum_to_one(fake_backend, rgb_fixture):
    service = InferenceService(fake_backend, temperature=2.0)
    prediction = service.predict(rgb_fixture)
    assert prediction.probability_ai + prediction.probability_real == pytest.approx(1.0)
    assert prediction.confidence == pytest.approx(max(prediction.probability_ai, prediction.probability_real))


def test_stability_reports_absolute_score_change(fake_backend, rgb_fixture):
    service = InferenceService(fake_backend, temperature=1.0)
    result = service.compare_transform(rgb_fixture, get_spec("jpeg_q30"), seed=17)
    assert result.condition_id == "jpeg_q30"
    assert result.absolute_change == pytest.approx(
        abs(result.transformed.probability_ai - result.clean.probability_ai)
    )
```

- [ ] **Step 2: Implement a backend protocol and shared service**

```python
class LogitBackend(Protocol):
    model_version: str
    predict_logit: Callable[[Image.Image], float]


class InferenceService:
    def __init__(self, backend: LogitBackend, temperature: float) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.backend = backend
        self.temperature = temperature

    def predict(self, image: Image.Image) -> Prediction:
        started = time.perf_counter()
        logit = self.backend.predict_logit(image.convert("RGB"))
        probability_ai = 1.0 / (1.0 + math.exp(-logit / self.temperature))
        probability_real = 1.0 - probability_ai
        return Prediction(
            probability_ai=probability_ai,
            probability_real=probability_real,
            confidence=max(probability_ai, probability_real),
            logit=logit,
            model_version=self.backend.model_version,
            inference_ms=(time.perf_counter() - started) * 1000.0,
        )

    def compare_transform(
        self, image: Image.Image, spec: TransformSpec, seed: int
    ) -> StabilityResult:
        clean = self.predict(image)
        transformed = self.predict(apply_transform(image, spec, seed))
        return StabilityResult(
            condition_id=spec.condition_id,
            clean=clean,
            transformed=transformed,
            absolute_change=abs(transformed.probability_ai - clean.probability_ai),
        )
```

The torch backend loads the selected checkpoint on `cuda` only when explicitly requested and available; otherwise it loads on CPU. The service applies calibration and measures wall-clock inference time.

- [ ] **Step 3: Run tests and commit**

Run: `python -m pytest tests/unit/inference/test_service.py tests/unit/inference/test_torch_backend.py -v`

Expected: PASS.

```bash
git add src/prooflens/inference tests/unit/inference
git commit -m "feat: add shared calibrated inference service"
```

### Task 13: ONNX export, CPU runtime, and OpenVINO smoke path

**Files:**

- Create: `src/prooflens/export/onnx_export.py`
- Create: `src/prooflens/export/openvino_export.py`
- Create: `src/prooflens/inference/onnx_backend.py`
- Create: `tests/integration/test_export_parity.py`
- Create: `tests/unit/export/test_openvino_export.py`

**Interfaces:**

- Consumes: Selected PyTorch detector and fixed parity batch.
- Produces: `model.onnx`, `export_report.json`, ONNX backend, optional OpenVINO compilation report.

- [ ] **Step 1: Write ONNX parity integration test with a tiny detector**

```python
def test_onnx_logits_match_pytorch(tmp_path, tiny_detector):
    sample = torch.randn(2, 3, 28, 28)
    onnx_path = export_onnx(tiny_detector.eval(), sample, tmp_path / "model.onnx")
    expected = tiny_detector(sample).logits.detach().numpy()
    actual = OnnxTensorBackend(onnx_path).predict_batch(sample.numpy())
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
```

- [ ] **Step 2: Implement the recommended Dynamo exporter**

```python
torch.onnx.export(
    LogitOnlyWrapper(model).eval(),
    (sample_pixels,),
    onnx_path,
    input_names=["pixel_values"],
    output_names=["logits"],
    dynamo=True,
    dynamic_shapes={"pixel_values": {0: torch.export.Dim("batch", min=1, max=32)}},
    opset_version=18,
)


class OnnxTensorBackend:
    def __init__(self, model_path: Path) -> None:
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def predict_batch(self, pixel_values: np.ndarray) -> np.ndarray:
        values = np.asarray(pixel_values, dtype=np.float32)
        return np.asarray(self.session.run(["logits"], {self.input_name: values})[0])


class OnnxLogitBackend:
    def __init__(self, model_path: Path, processor, model_version: str) -> None:
        self.tensor_backend = OnnxTensorBackend(model_path)
        self.processor = processor
        self.model_version = model_version

    def predict_logit(self, image: Image.Image) -> float:
        pixel_values = self.processor(images=[image], return_tensors="np")["pixel_values"]
        return float(self.tensor_backend.predict_batch(pixel_values).reshape(-1)[0])
```

Use `onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])` for the required laptop backend. Verify at least 32 representative parity images and reject publication when maximum absolute probability difference exceeds `1e-4`.

- [ ] **Step 3: Implement optional OpenVINO compilation without making it a release gate**

```python
def compile_openvino(onnx_path: Path, device: str = "AUTO"):
    import openvino as ov
    core = ov.Core()
    return core.compile_model(onnx_path, device)
```

The OpenVINO test uses `pytest.importorskip("openvino")`. The release report records success, device name, and one smoke prediction. A failure preserves the valid ONNX CPU artifact.

- [ ] **Step 4: Run export tests and commit**

Run: `python -m pytest tests/integration/test_export_parity.py tests/unit/export/test_openvino_export.py -v`

Expected: ONNX parity PASS; OpenVINO PASS when installed or SKIP when absent.

```bash
git add src/prooflens/export src/prooflens/inference/onnx_backend.py tests/integration/test_export_parity.py tests/unit/export
git commit -m "feat: add ONNX and optional OpenVINO export"
```

### Task 14: Gradio demonstration application

**Files:**

- Create: `src/prooflens/web/app.py`
- Create: `tests/integration/test_web_service.py`

**Interfaces:**

- Consumes: `InferenceService`, uploaded PIL image, canonical condition ID.
- Produces: Gradio Blocks app and `analyze_upload()` response tuple.

- [ ] **Step 1: Write upload analysis tests independent of browser launch**

```python
def test_analyze_upload_returns_clean_and_transformed_results(fake_service, rgb_fixture):
    result = analyze_upload(rgb_fixture, "jpeg_q30", fake_service)
    assert result.clean_image.size == rgb_fixture.size
    assert result.transformed_image.size == rgb_fixture.size
    assert result.summary["condition"] == "jpeg_q30"
    assert "probability_ai" in result.summary["clean"]


def test_analyze_upload_rejects_missing_image(fake_service):
    with pytest.raises(UserInputError, match="Upload an image"):
        analyze_upload(None, "jpeg_q30", fake_service)
```

- [ ] **Step 2: Implement Gradio Blocks with the shared service**

Create an image input, transform dropdown, analyze button, clean and transformed image outputs, probability labels, JSON stability summary, model-information accordion, and limitations panel. Catch `UserInputError` and return a readable Gradio error without terminating the server.

```python
@dataclass(frozen=True)
class UploadAnalysis:
    clean_image: Image.Image
    transformed_image: Image.Image
    summary: dict[str, object]

    def as_outputs(
        self,
    ) -> tuple[Image.Image, Image.Image, dict[str, float], dict[str, float], dict[str, object]]:
        clean = self.summary["clean"]
        transformed = self.summary["transformed"]
        clean_probabilities = {
            "AI-generated": clean["probability_ai"], "Authentic": clean["probability_real"]
        }
        transformed_probabilities = {
            "AI-generated": transformed["probability_ai"],
            "Authentic": transformed["probability_real"],
        }
        return (
            self.clean_image,
            self.transformed_image,
            clean_probabilities,
            transformed_probabilities,
            self.summary,
        )


def analyze_upload(
    image: Image.Image | None,
    condition_id: str,
    service: InferenceService,
) -> UploadAnalysis:
    if image is None:
        raise UserInputError("Upload an image before analysis")
    clean_image = image.convert("RGB")
    spec = get_spec(condition_id)
    transformed_image = apply_transform(clean_image, spec, seed=17)
    stability = service.compare_transform(clean_image, spec, seed=17)
    return UploadAnalysis(
        clean_image=clean_image,
        transformed_image=transformed_image,
        summary={
            "condition": condition_id,
            "clean": asdict(stability.clean),
            "transformed": asdict(stability.transformed),
            "absolute_change": stability.absolute_change,
        },
    )


def create_app(service: InferenceService) -> gr.Blocks:
    with gr.Blocks(title="ProofLens") as app:
        image = gr.Image(type="pil", label="Image")
        condition = gr.Dropdown(
            choices=[spec.condition_id for spec in canonical_specs()],
            value="jpeg_q30",
            label="Robustness check",
        )
        analyze = gr.Button("Analyze")
        clean_output = gr.Image(label="Clean")
        transformed_output = gr.Image(label="Transformed")
        clean_probabilities = gr.Label(label="Clean probabilities")
        transformed_probabilities = gr.Label(label="Transformed probabilities")
        summary = gr.JSON(label="Prediction stability")
        analyze.click(
            fn=lambda uploaded, selected: analyze_upload(uploaded, selected, service).as_outputs(),
            inputs=[image, condition],
            outputs=[
                clean_output,
                transformed_output,
                clean_probabilities,
                transformed_probabilities,
                summary,
            ],
        )
    return app
```

- [ ] **Step 3: Run service tests and a manual local smoke check**

Run: `python -m pytest tests/integration/test_web_service.py -v`

Expected: PASS.

Run: `python -m prooflens.cli app --backend torch --checkpoint artifacts/checkpoints/final.pt`

Expected: Gradio reports a local URL and accepts one valid image without a Python traceback.

- [ ] **Step 4: Commit the app**

```bash
git add src/prooflens/web tests/integration/test_web_service.py
git commit -m "feat: add local robustness demonstration"
```

### Task 15: Unified CLI, experiment configurations, and miniature workflow

**Files:**

- Create: `src/prooflens/cli.py`
- Create: `configs/experiments/e1_last2.yaml`
- Create: `configs/experiments/e2_augmented.yaml`
- Create: `configs/experiments/e3_consistency.yaml`
- Create: `configs/experiments/e4_hard_mining.yaml`
- Create: `tests/fixtures/make_fixture_data.py`
- Create: `scripts/reproduce_small.py`
- Create: `tests/integration/test_small_workflow.py`

**Interfaces:**

- Consumes: YAML configurations and local paths.
- Produces: `python -m prooflens.cli` subcommands for `acquire`, `manifest`, `audit`, `split`, `train`, `evaluate`, `select`, `calibrate`, `report`, `export`, and `app`.

- [ ] **Step 1: Write CLI help and miniature workflow tests**

```python
def test_cli_lists_required_commands():
    result = subprocess.run(
        [sys.executable, "-m", "prooflens.cli", "--help"],
        text=True, capture_output=True, check=True,
    )
    for command in ("acquire", "manifest", "audit", "split", "train", "evaluate", "select", "calibrate", "report", "export", "app"):
        assert command in result.stdout


def test_small_reproduction_creates_required_artifacts(tmp_path):
    result = reproduce_small(tmp_path)
    assert result.checkpoint.exists()
    assert result.predictions.exists()
    assert result.metrics.exists()
    assert result.robustness_markdown.exists()
```

- [ ] **Step 2: Implement argparse command dispatch with no duplicated business logic**

Each command parses paths and overrides, then calls the corresponding package function. Return exit code `2` for user configuration errors, `3` for data-integrity failures, and `4` for model or export failures.

```python
COMMANDS = (
    "acquire", "manifest", "audit", "split", "train", "evaluate",
    "select", "calibrate", "report", "export", "app",
)


COMMAND_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "acquire": run_acquire_cli,
    "manifest": run_manifest_cli,
    "audit": run_audit_cli,
    "split": run_split_cli,
    "train": run_train_cli,
    "evaluate": run_evaluate_cli,
    "select": run_select_cli,
    "calibrate": run_calibrate_cli,
    "report": run_report_cli,
    "export": run_export_cli,
    "app": run_app_cli,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prooflens")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        child = subparsers.add_parser(command)
        child.set_defaults(handler=COMMAND_HANDLERS[command])

    acquire = subparsers.choices["acquire"]
    acquire.add_argument("--config", type=Path, required=True)
    acquire.add_argument("--output", type=Path, required=True)

    manifest = subparsers.choices["manifest"]
    manifest.add_argument("--config", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)

    audit = subparsers.choices["audit"]
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)

    split = subparsers.choices["split"]
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=17)

    train = subparsers.choices["train"]
    train_source = train.add_mutually_exclusive_group(required=True)
    train_source.add_argument("--config", type=Path)
    train_source.add_argument("--config-from-selection", type=Path)
    train.add_argument("--seed", type=int)
    train.add_argument("--output", type=Path)

    evaluate = subparsers.choices["evaluate"]
    evaluate_source = evaluate.add_mutually_exclusive_group(required=True)
    evaluate_source.add_argument("--run", type=Path)
    evaluate_source.add_argument("--selection", type=Path)
    evaluate.add_argument("--suite", choices=("clean", "clean-robust-generator"), required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")

    select = subparsers.choices["select"]
    select.add_argument("--runs", type=Path, nargs="+", required=True)
    select.add_argument("--output", type=Path, default=Path("artifacts/selection.json"))

    calibrate = subparsers.choices["calibrate"]
    calibrate.add_argument("--selection", type=Path, required=True)
    calibrate.add_argument("--split", choices=("validation",), default="validation")
    calibrate.add_argument("--output", type=Path, required=True)

    report = subparsers.choices["report"]
    report.add_argument("--selection", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)

    export = subparsers.choices["export"]
    export.add_argument("--selection", type=Path, required=True)
    export.add_argument("--format", choices=("onnx", "openvino"), default="onnx")
    export.add_argument("--verify", type=int, default=32)
    export.add_argument("--output", type=Path, required=True)

    app = subparsers.choices["app"]
    app.add_argument("--backend", choices=("torch", "onnx"), default="onnx")
    app.add_argument("--checkpoint", type=Path)
    app.add_argument("--model", type=Path)
    app.add_argument("--calibration", type=Path)
    return parser


def dispatch(args: argparse.Namespace) -> int:
    try:
        return int(args.handler(args))
    except UserInputError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2
    except DataIntegrityError as error:
        print(f"data integrity error: {error}", file=sys.stderr)
        return 3
    except (TrainingError, ExportError) as error:
        print(f"model error: {error}", file=sys.stderr)
        return 4


def main() -> None:
    raise SystemExit(dispatch(build_parser().parse_args()))
```

- [ ] **Step 3: Lock experiment differences in YAML**

`E0` uses `stage: head`, clean batches, and consistency weights zero. `E1` uses `stage: last2`. `E2` enables paired transforms with consistency weights zero. `E3` sets prediction weight `0.25` and feature weight `0.10`. `E4` inherits E3 settings and enables hard mining with `candidate_count: 3`.

Every YAML file contains the complete schema rather than relying on an implicit inheritance mechanism. Use these output directories and distinguishing values:

```yaml
# e1_last2.yaml distinguishing values
model: {name: facebook/dinov2-base, stage: last2}
transforms: {enabled: false, hard_mining: false, candidate_count: 3, exploration_probability: 0.20}
loss: {clean_bce: 1.0, transformed_bce: 0.0, prediction_consistency: 0.0, feature_consistency: 0.0}
output_dir: artifacts/runs/e1

# e2_augmented.yaml distinguishing values
model: {name: facebook/dinov2-base, stage: last2}
transforms: {enabled: true, hard_mining: false, candidate_count: 3, exploration_probability: 0.20}
loss: {clean_bce: 1.0, transformed_bce: 1.0, prediction_consistency: 0.0, feature_consistency: 0.0}
output_dir: artifacts/runs/e2

# e3_consistency.yaml distinguishing values
transforms: {enabled: true, hard_mining: false, candidate_count: 3, exploration_probability: 0.20}
loss: {clean_bce: 1.0, transformed_bce: 1.0, prediction_consistency: 0.25, feature_consistency: 0.10}
output_dir: artifacts/runs/e3

# e4_hard_mining.yaml distinguishing values
transforms: {enabled: true, hard_mining: true, candidate_count: 3, exploration_probability: 0.20}
loss: {clean_bce: 1.0, transformed_bce: 1.0, prediction_consistency: 0.25, feature_consistency: 0.10}
output_dir: artifacts/runs/e4
```

- [ ] **Step 4: Implement the network-free miniature workflow**

The fixture generator creates deterministic geometric RGB images for both labels. The miniature workflow uses a tiny DINO configuration initialized from a fixed seed, exercises every primary subsystem, and finishes on CPU without downloading data or model weights.

```python
@dataclass(frozen=True)
class ReproductionResult:
    checkpoint: Path
    predictions: Path
    metrics: Path
    robustness_markdown: Path

    @classmethod
    def from_artifacts(cls, training, predictions: Path, report) -> "ReproductionResult":
        return cls(
            checkpoint=training.best_checkpoint,
            predictions=predictions,
            metrics=report.metrics_json,
            robustness_markdown=report.robustness_markdown,
        )


def reproduce_small(output_dir: Path) -> ReproductionResult:
    fixture_root = make_fixture_data(output_dir / "fixture", per_class=8, seed=17)
    manifest = build_fixture_manifest(fixture_root, output_dir / "manifest.parquet")
    split = build_fixture_split(manifest, output_dir / "split.parquet", seed=17)
    training = train_fixture_model(split, output_dir / "run", seed=17)
    predictions = evaluate_fixture_model(training.best_checkpoint, split, output_dir / "predictions.parquet")
    report = build_fixture_report(predictions, output_dir / "report")
    return ReproductionResult.from_artifacts(training, predictions, report)
```

- [ ] **Step 5: Run workflow tests and commit**

Run: `python -m pytest tests/integration/test_small_workflow.py -v`

Run: `python scripts/reproduce_small.py --output artifacts/smoke`

Expected: Both commands PASS and the script prints paths for checkpoint, predictions, metrics, and report.

```bash
git add src/prooflens/cli.py configs/experiments tests/fixtures scripts/reproduce_small.py tests/integration/test_small_workflow.py
git commit -m "feat: connect the end-to-end experiment workflow"
```

### Task 16: Documentation, notices, CI, and release checks

**Files:**

- Create: `README.md`
- Create: `LICENSE`
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `docs/datasets.md`
- Create: `docs/model-card.md`
- Create: `docs/devpost-draft.md`
- Create: `docs/video-script.md`
- Create: `scripts/release_check.py`
- Create: `tests/unit/test_release_check.py`
- Create: `.github/workflows/ci.yml`

**Interfaces:**

- Consumes: Final commands, licence registry, metric artifacts, model and dataset identifiers.
- Produces: Complete public submission documentation and automated release gate.

- [ ] **Step 1: Write release-check tests**

```python
def test_release_check_rejects_secret_like_files(tmp_path):
    (tmp_path / ".env").write_text("HF_TOKEN=secret")
    result = run_release_check(tmp_path)
    assert not result.ok
    assert any(".env" in error for error in result.errors)


def test_release_check_requires_submission_documents(project_fixture):
    (project_fixture / "docs/model-card.md").unlink()
    result = run_release_check(project_fixture)
    assert not result.ok
    assert any("docs/model-card.md" in error for error in result.errors)
```

- [ ] **Step 2: Write documentation using verified commands**

The README must include Windows and Linux setup, manual and automated dataset paths, E0 through E4 commands, evaluation aggregation, export, Gradio launch, artifact layout, expected failure messages, licence notes, and a one-command miniature reproduction. Every documented command must be exercised before publication.

`THIRD_PARTY_NOTICES.md` must identify DINOv2 as Apache-2.0, SID-Set as CC-BY-4.0 with source-material attribution requirements, CIFAKE as MIT with required citations, and WildFake with its official paper and acquisition source while marking its licence as requiring verification before redistribution.

- [ ] **Step 3: Implement release checks and CPU CI**

The release checker scans tracked paths for `.env`, private keys, common token patterns, raw data folders, oversized files, missing documents, missing licences, and missing model-card fields. CI runs Ruff, unit tests, the miniature workflow, and ONNX parity on both `windows-latest` and `ubuntu-latest` using Python 3.11.

```python
@dataclass(frozen=True)
class ReleaseCheckResult:
    ok: bool
    errors: tuple[str, ...]


REQUIRED_DOCUMENTS = (
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md",
    "docs/datasets.md", "docs/model-card.md",
    "docs/devpost-draft.md", "docs/video-script.md",
)


def run_release_check(root: Path) -> ReleaseCheckResult:
    errors: list[str] = []
    for relative in REQUIRED_DOCUMENTS:
        if not (root / relative).is_file():
            errors.append(f"missing required document: {relative}")

    forbidden_names = {".env", "id_rsa", "id_ed25519"}
    token_pattern = re.compile(
        r"(?i)(hf_[a-z0-9]{20,}|github_pat_[a-z0-9_]{20,}|api[_-]?key\s*[=:]\s*\S+)"
    )
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts or ".venv" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if path.name in forbidden_names or path.suffix in {".pem", ".key"}:
            errors.append(f"secret-like file must not be published: {relative}")
        if path.stat().st_size > 100 * 1024 * 1024:
            errors.append(f"file exceeds 100 MiB release limit: {relative}")
        if relative.startswith("data/raw/"):
            errors.append(f"raw dataset file must not be published: {relative}")
        if path.stat().st_size <= 2 * 1024 * 1024:
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if token_pattern.search(text):
                errors.append(f"possible credential in: {relative}")

    model_card = root / "docs/model-card.md"
    if model_card.is_file():
        text = model_card.read_text(encoding="utf-8").lower()
        for field in ("intended use", "limitations", "datasets", "metrics", "ethical"):
            if field not in text:
                errors.append(f"model card missing field: {field}")
    return ReleaseCheckResult(ok=not errors, errors=tuple(sorted(set(errors))))
```

- [ ] **Step 4: Run the complete release gate**

Run: `python -m ruff check src tests scripts`

Run: `python -m pytest -v`

Run: `python scripts/reproduce_small.py --output artifacts/release-smoke`

Run: `python scripts/release_check.py --root .`

Expected: Ruff PASS, all required tests PASS, optional OpenVINO tests PASS or SKIP, miniature workflow PASS, release check reports `ok: true`.

- [ ] **Step 5: Commit release assets**

```bash
git add README.md LICENSE THIRD_PARTY_NOTICES.md docs scripts/release_check.py tests/unit/test_release_check.py .github/workflows/ci.yml
git commit -m "docs: add reproducible release and submission assets"
```

### Task 17: Execute E0 through E4 and freeze the final candidate

**Files:**

- Modify: `configs/experiments/e0_frozen.yaml`
- Modify: `configs/experiments/e1_last2.yaml`
- Modify: `configs/experiments/e2_augmented.yaml`
- Modify: `configs/experiments/e3_consistency.yaml`
- Modify: `configs/experiments/e4_hard_mining.yaml`
- Create: `docs/reports/experiment-summary.md`
- Create: `docs/reports/final-robustness.md`
- Create: `docs/reports/error-analysis.md`

**Interfaces:**

- Consumes: Validated primary manifest, fixed split, E0 through E4 configurations.
- Produces: Recorded run IDs, selected final checkpoint, calibration, exported model, and report assets.

- [ ] **Step 1: Freeze one manifest and split fingerprint**

Run: `python -m prooflens.cli manifest --config configs/data/primary.yaml --output artifacts/manifests/primary.parquet`

Run: `python -m prooflens.cli audit --manifest artifacts/manifests/primary.parquet --output artifacts/reports/data-audit`

Run: `python -m prooflens.cli split --manifest artifacts/manifests/primary.parquet --output artifacts/manifests/primary-split.parquet --seed 17`

Expected: Audit contains both labels, split command reports zero leakage, and the split manifest SHA-256 is recorded.

- [ ] **Step 2: Run E0 and E1 on identical splits**

Run: `python -m prooflens.cli train --config configs/experiments/e0_frozen.yaml`

Run: `python -m prooflens.cli evaluate --run artifacts/runs/e0 --suite clean-robust-generator --split validation`

Run: `python -m prooflens.cli train --config configs/experiments/e1_last2.yaml`

Run: `python -m prooflens.cli evaluate --run artifacts/runs/e1 --suite clean-robust-generator --split validation`

Expected: Each run emits a checkpoint, prediction Parquet, metrics JSON, and robustness Markdown using the same manifest and split hashes.

- [ ] **Step 3: Run E2, E3, and E4 without changing the data split**

Run: `python -m prooflens.cli train --config configs/experiments/e2_augmented.yaml`

Run: `python -m prooflens.cli evaluate --run artifacts/runs/e2 --suite clean-robust-generator --split validation`

Run: `python -m prooflens.cli train --config configs/experiments/e3_consistency.yaml`

Run: `python -m prooflens.cli evaluate --run artifacts/runs/e3 --suite clean-robust-generator --split validation`

Run: `python -m prooflens.cli train --config configs/experiments/e4_hard_mining.yaml`

Run: `python -m prooflens.cli evaluate --run artifacts/runs/e4 --suite clean-robust-generator --split validation`

Record clean AUC, macro robust AUC, composite score, worst family, worst condition, unseen-generator AUC, training time, and inference time in `docs/reports/experiment-summary.md`.

- [ ] **Step 4: Select the leading configuration and repeat it on seeds 29 and 41**

Run: `python -m prooflens.cli select --runs artifacts/runs/e0 artifacts/runs/e1 artifacts/runs/e2 artifacts/runs/e3 artifacts/runs/e4 --output artifacts/selection.json`

Run: `python -m prooflens.cli train --config-from-selection artifacts/selection.json --seed 29 --output artifacts/runs/selected-seed29`

Run: `python -m prooflens.cli evaluate --run artifacts/runs/selected-seed29 --suite clean-robust-generator --split validation`

Run: `python -m prooflens.cli train --config-from-selection artifacts/selection.json --seed 41 --output artifacts/runs/selected-seed41`

Run: `python -m prooflens.cli evaluate --run artifacts/runs/selected-seed41 --suite clean-robust-generator --split validation`

Use the three seeds to estimate mean and standard deviation for clean, macro robust, composite, and worst-family validation AUC. Keep the provisionally selected configuration unless its repeat composite varies by more than 0.02 absolute or worst-family AUC varies by more than 0.03. If that stability gate fails, repeat the runner-up configuration on seeds 29 and 41 and choose the higher mean validation composite, using mean worst-family AUC as the tie-break. Choose the deployment seed whose validation composite is closest to its configuration mean, then update `artifacts/selection.json`. Do not inspect `test` or `generator_test` predictions during this decision.

- [ ] **Step 5: Select, calibrate, export, and verify the final checkpoint**

Run: `python -m prooflens.cli calibrate --selection artifacts/selection.json --split validation --output artifacts/export/calibration.json`

Run: `python -m prooflens.cli export --selection artifacts/selection.json --format onnx --verify 32 --output artifacts/export/model.onnx`

Run: `python -m prooflens.cli evaluate --selection artifacts/selection.json --suite clean-robust-generator --split test`

Run: `python -m prooflens.cli report --selection artifacts/selection.json --output artifacts/reports/final`

Expected: The selected run follows the validation composite and tie-break rules, calibration uses validation only, ONNX parity is within `1e-4`, final test evaluation runs only after the selection is frozen, and final reports contain every required condition and error category.

- [ ] **Step 6: Record measured conclusions and commit report sources**

Commit Markdown reports and small plots only. Keep checkpoints, raw predictions, thumbnails derived from restricted data, and large binary exports in release storage rather than git.

```bash
git add docs/reports configs/experiments
git commit -m "docs: record robust detector experiments"
```

### Task 18: Final laptop and clean-checkout acceptance

**Files:**

- Modify: `README.md`
- Modify: `docs/model-card.md`
- Modify: `docs/devpost-draft.md`
- Modify: `docs/video-script.md`
- Create: `docs/reports/acceptance-report.md`

**Interfaces:**

- Consumes: Final checkpoint, ONNX model, calibration, final reports, clean repository checkout.
- Produces: Signed-off acceptance report and submission-ready repository.

- [ ] **Step 1: Test CPU ONNX inference on the target Intel laptop**

Run: `python -m prooflens.cli app --backend onnx --model artifacts/export/model.onnx --calibration artifacts/export/calibration.json`

Expected: The app launches without CUDA, accepts a valid image, returns probabilities, and compares one selected transformation.

- [ ] **Step 2: Run a clean-checkout miniature reproduction**

Create a new directory outside the development checkout, clone the repository, create a Python 3.11 virtual environment, install `.[dev]`, and run:

```bash
python scripts/reproduce_small.py --output artifacts/clean-check
python scripts/release_check.py --root .
```

Expected: Miniature workflow PASS and release check reports `ok: true`.

- [ ] **Step 3: Verify submission assets against judging categories**

Record evidence for technical execution, innovation and problem insight, impact and relevance, feasibility and practicality, and presentation and communication in `docs/reports/acceptance-report.md`. Include direct paths to the robustness table, error analysis, model card, demo command, and video script.

- [ ] **Step 4: Run the final verification suite**

Run: `python -m ruff check src tests scripts`

Run: `python -m pytest -v`

Run: `python scripts/release_check.py --root .`

Expected: All required checks PASS. OpenVINO may SKIP or fail without blocking release only when ONNX CPU inference passes.

- [ ] **Step 5: Commit the acceptance record**

```bash
git add README.md docs/model-card.md docs/devpost-draft.md docs/video-script.md docs/reports/acceptance-report.md
git commit -m "docs: finalize hackathon acceptance evidence"
```

---

## Optional Module Gate

Frequency and authentic-manifold branches are deliberately outside this committed plan. Create a separate approved design and implementation plan for either branch only when all of these conditions hold:

1. Tasks 1 through 16 pass.
2. E0 through E4 results are recorded.
3. The final-day submission work has not begun.
4. At least eight development hours remain before the feature freeze.
5. A small prototype improves the composite validation score by at least 0.005 absolute and does not reduce worst-family AUC by more than 0.005.

## Plan Completion Evidence

The implementation is complete when the Task 18 verification commands pass, the laptop runs CPU ONNX inference, the final robustness and error-analysis reports exist, and the public repository contains the documented reproducibility and submission assets.
