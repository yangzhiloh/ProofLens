# Task 9 acceptance report

Date: 2026-08-31  
Branch: `codex/task9-acceptance`  
Task 8 export commit: `d881cd3`  
Windows launch fix commit: `1e002e9`

## Laptop acceptance

The parity-verified E2 ONNX model was loaded on a Windows CPU and exercised through the live
Gradio HTTP API with a held-out valid image and the canonical `jpeg_q30` transformation.

| Check | Result |
| --- | --- |
| Clean prediction | Authentic 0.904261; AI-generated 0.095739 |
| Transformed prediction | Authentic 0.893660; AI-generated 0.106340 |
| Absolute AI probability change | 0.010600 |
| Stability message | Stable under transformation |
| Clean model inference | 323.58 ms |
| Transformed model inference | 315.93 ms |
| Two-pass model inference | 639.5 ms |
| End-to-end HTTP round trip | 2846.35 ms |
| Model provenance | `prooflens-e2-onnx` |
| Preprocessing provenance | `dinov2-base-224-v1` |

The app reported both calibrated probabilities, both verdicts, the selected condition, the
probability change, timing, model version, and preprocessing version. Visual browser automation
could not connect to the host loopback service, so acceptance used the live Gradio endpoint
rather than a scripted browser click-through.

## Clean-checkout acceptance

A fresh clone of the pushed branch was created at
`C:\Users\Yang Jie\AppData\Local\Temp\prooflens-task9-clean-check`. A clean CPython 3.11.15
environment was synchronized with `uv sync --locked --extra dev`. The following commands passed:

```text
uv run --locked --extra dev python scripts/reproduce_small.py --output artifacts/clean-check
uv run --locked --extra dev python scripts/release_check.py --root .
```

The reproduction generated a checkpoint, predictions, metrics, and robustness report without
using the primary dataset or its weights. The release scanner returned `release check: OK`.

## Judging evidence map

| Category | Evidence |
| --- | --- |
| Technical execution | [Experiment summary](experiment-summary.md), [final robustness](final-robustness.md), and ONNX parity |
| Innovation and problem insight | Transformation-robust training sequence and hard-transform experiment in the [experiment summary](experiment-summary.md) |
| Impact and relevance | Intended use and responsible-use boundaries in the [model card](../model-card.md) |
| Feasibility and practicality | CPU ONNX acceptance above and clean-checkout reproduction |
| Presentation and communication | [README](../../README.md), [video script](../video-script.md), and [Devpost draft](../devpost-draft.md) |

## Sign-off

- CPU ONNX load and inference: PASS
- Valid image analysis: PASS
- Clean/transformed comparison: PASS
- Clean-checkout locked installation and reproduction: PASS
- Release scanner in clean checkout: PASS
- Final repository suite: PASS (440 tests passed, 1 optional OpenVINO test skipped, Ruff passed,
  and release check passed)

The model-artifact bundle is hosted at
`https://github.com/yangzhiloh/Tiktoky/releases/tag/prooflens-e2-rc1`. The remaining human-only
actions are recording/uploading the demonstration, reviewing the final publication copy, and
submitting it through the destination account.
