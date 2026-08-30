# Task 8 training preflight

This package prepares production training without downloading a dataset or starting a training
run. Execute it with:

```powershell
.\scripts\prooflens.ps1 preflight
```

```bash
bash scripts/prooflens.sh preflight
```

The report is written to `artifacts/preflight/task8.json`. It records disk capacity, CUDA and
VRAM availability, a conservative batch/accumulation recommendation, dataset structure,
credential presence, licence gates, experiment configuration validity, robustness transforms,
and resume support. Missing datasets are warnings; unresolved licence terms are blockers.

## Human approval gates

The source registry is `configs/data/task8_sources.yaml`.

- SID-Set: its Hugging Face source card identifies CC BY 4.0, requires attribution, lists labels
  0 authentic, 1 fully synthetic, and 2 tampered, and reports about 140 GB for the full hosted
  dataset. ProofLens streams a pinned revision and selects 10,000 each from labels 0 and 1 only.
- CIFAKE: open the Kaggle dataset page while signed in, record the licence shown for the exact
  downloadable version, accept any terms, and only then change `approved_for_acquisition` to
  `true`. It remains supplemental stress-only because its low-resolution/source characteristics
  can create shortcuts in primary training.
- WildFake: use the translation button on the ModelScope page, save the translated licence and
  usage terms with the experiment records, confirm research/training and redistribution rights,
  and only then change `approved_for_acquisition` to `true`. Its local export must contain
  `real/` and at least three direct generator-family directories below `fake/`.

Approval means permission to acquire and use the dataset for this project; it does not imply that
raw images may be committed or redistributed. `data/` and `artifacts/` remain ignored.

## Dataset and storage preparation

Reserve at least 100 GiB free before acquisition. The actual requirement depends on the WildFake
export and provider caches. Configure Kaggle credentials without committing them. The preflight
only checks for `KAGGLE_API_TOKEN` or `~/.kaggle/kaggle.json`; it never reads or prints a secret.

Once the human gates are cleared, run the commands listed in the preflight JSON in order. SID-Set
has an automated pinned acquisition command. CIFAKE and WildFake remain manual imports so their
terms and exact local copies are consciously reviewed.

## Pilot before full training

After the canonical split exists, derive a deterministic pilot manifest:

```text
uv run --locked --extra dev python scripts/task8_preflight.py --source-manifest artifacts/manifests/primary-split.parquet
uv run --locked --extra dev python -m prooflens.cli train --config configs/experiments/e0_pilot.yaml
```

The pilot takes at most 32 rows per label from each assigned split, keeps the existing split
assignments, trains the frozen-head stage for one epoch with batch size 4, and writes to a separate
run directory. It is a pipeline check, not a reported experiment.

## Full experiment safety

- CUDA automatically enables mixed precision; CPU leaves it disabled.
- Every epoch writes a recoverable checkpoint and appends `history.jsonl` metrics.
- Early stopping is configured for E0 through E4.
- Resume an interrupted run with `prooflens train --config <config> --resume-from <checkpoint>`.
- Start with the preflight batch recommendation. CUDA out-of-memory errors report a smaller batch
  size; preserve effective batch size with gradient accumulation.
- The canonical split groups exact/perceptual duplicates and source identities before training.
- E0 through E4 share the same split. Select on validation data, calibrate on validation data, and
  evaluate the selected model on test data only after selection.

## Post-training command order

Run validation evaluation for E0 through E4, select the winner, calibrate it, run canonical and
supplemental stress evaluation, produce reports/error analysis, and publish ONNX only after the
32-sample parity gate passes. The existing README contains the exact CLI commands and artifact
layout.
