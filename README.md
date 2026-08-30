# ProofLens

ProofLens is a robustness-oriented research demo for scoring whether an image is authentic or
AI-generated. It compares a clean prediction with a transformed version of the same image and
reports how stable the score is.

> [!IMPORTANT]
> ProofLens scores are not forensic proof. The repository can generate a small fixture model for
> testing the workflow, but that model is not a production detector and must not be presented as
> one.

## Run the local fixture demo

The fixture workflow is deterministic, uses generated images, and does not download a model or
dataset. It verifies training, calibration, ONNX export, CPU parity, and the Gradio interface.

### Windows PowerShell

From the repository directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe scripts/reproduce_small.py --output artifacts/demo --experiment e3 --publish-demo-artifacts
.\.venv\Scripts\python.exe -m prooflens.cli app --backend onnx --model artifacts/demo/export/model.onnx --calibration artifacts/demo/export/calibration.json
```

If `python` and the Windows `py` launcher are unavailable inside Codex, use the bundled runtime for
the first command:

```powershell
& "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" -m venv .venv
```

Open <http://127.0.0.1:7860> after Gradio prints the local address. Stop the server with `Ctrl+C`.

### Linux or macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python scripts/reproduce_small.py --output artifacts/demo --experiment e3 --publish-demo-artifacts
.venv/bin/python -m prooflens.cli app --backend onnx --model artifacts/demo/export/model.onnx --calibration artifacts/demo/export/calibration.json
```

## Generated demo artifacts

The miniature publication command creates:

```text
artifacts/demo/
├── selection.json
├── export/
│   ├── calibration.json
│   ├── artifact_manifest.json
│   ├── export_report.json
│   └── model.onnx
├── run/
│   ├── config.yaml
│   ├── predictions-validation.parquet
│   └── checkpoints/best.pt
└── report/
    ├── metrics.json
    ├── robustness.md
    └── auc.png
```

`selection.json` identifies these files as `deterministic-fixture-demo`.
`artifact_manifest.json` binds the model, calibration, preprocessing version, and SHA-256 hashes.
The app validates this sidecar and chooses fixture or DINOv2 preprocessing automatically.

## Production artifact workflow

A real release requires licensed datasets, completed E0 through E4 runs, and a trained DINOv2
checkpoint. Those inputs are intentionally not included in this clone.

After the training team provides `artifacts/runs/e0` through `artifacts/runs/e4`, run:

```powershell
.\.venv\Scripts\python.exe -m prooflens.cli select --runs artifacts/runs/e0 artifacts/runs/e1 artifacts/runs/e2 artifacts/runs/e3 artifacts/runs/e4 --output artifacts/selection.json
.\.venv\Scripts\python.exe -m prooflens.cli calibrate --selection artifacts/selection.json --split validation --output artifacts/export/calibration.json
.\.venv\Scripts\python.exe -m prooflens.cli export --selection artifacts/selection.json --format onnx --verify 32 --output artifacts/export/model.onnx
.\.venv\Scripts\python.exe -m prooflens.cli evaluate --selection artifacts/selection.json --suite clean-robust-generator --split test
.\.venv\Scripts\python.exe -m prooflens.cli report --selection artifacts/selection.json --output artifacts/reports/final
```

Launch a production ONNX export without the fixture flag:

```powershell
.\.venv\Scripts\python.exe -m prooflens.cli app --backend onnx --model artifacts/export/model.onnx --calibration artifacts/export/calibration.json
```

Model selection and calibration use validation data only. Test and generator-holdout predictions
must not influence checkpoint selection, temperature fitting, or threshold selection.

## What belongs in Git

The entire `artifacts/` directory is ignored to prevent datasets, checkpoints, predictions, and
large model binaries from entering normal commits.

For a production release:

- Commit source code, configurations, documentation, small Markdown reports, and non-sensitive
  plots.
- Publish `model.onnx` and PyTorch checkpoints as GitHub Release assets or in approved model
  storage.
- Publish `selection.json`, `calibration.json`, and `export_report.json` beside the model so their
  provenance remains together.
- Publish `artifact_manifest.json` with the bundle; automatic launch rejects missing, mixed, or
  hash-mismatched artifacts.
- Never publish source datasets, restricted thumbnails, credentials, or raw predictions unless
  their licences and privacy requirements explicitly permit it.
- Record the release asset URL and SHA-256 checksum in the release notes.

Do not use `git add -f artifacts/...` for the fixture bundle. Regenerate it with
`scripts/reproduce_small.py` instead.

## Troubleshooting

- `py is not recognized`: use `python`, or use the bundled Codex Python command shown above.
- `-e option requires 1 argument`: include the final dot in `pip install -e .`.
- `No module named prooflens`: the editable installation did not complete; rerun
  `.\.venv\Scripts\python.exe -m pip install -e .` and wait for `Successfully installed`.
- Installation appears frozen at `Installing collected packages`: PyTorch and its dependencies can
  take several minutes to unpack on Windows. Wait for the PowerShell prompt to return.
- Fixture results stay near 50%: the fixture is a workflow test, not a trained real-world detector.
- `automatic preprocessing requires artifact_manifest.json`: regenerate the bundle, or use an
  explicit `--preprocessing dinov2` only after independently verifying a legacy model.
- A production launch downloads DINOv2 processor metadata on first use. The fixture launch is
  offline after dependencies are installed.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
```

The same Ruff check and test suite run automatically in GitHub Actions for every pull request.
Tests run on both supported Python versions, 3.11 and 3.12; superseded runs are cancelled.

The required exported runtime is ONNX Runtime on CPU. OpenVINO acceleration is optional.
