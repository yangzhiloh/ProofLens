# SDD ledger: plan: docs/superpowers/plans/2026-08-28-robust-ai-image-detector-implementation.md

Workspace: C:\Users\Loh Yang Zhi\Documents\ChatGPT\Tiktoky\.worktrees\prooflens-implementation
Branch: codex/prooflens-implementation
Merge base: 9b16823
Correct delivery path: C:\Users\Loh Yang Zhi\Documents\Projects\Tiktoky

## Setup rulings

Ruling: Emulated `sdd-workspace` in PowerShell because the bundled Bash script fails on its CRLF line endings under WSL; preserved the exact plan-scoped `.superpowers/sdd/<plan-basename>/` layout and self-ignoring file; cost if wrong: review artifacts may need relocation, while git history remains unaffected.

Ruling: Use the isolated worktree as the implementation source of truth and synchronize reviewed committed files to `Documents\Projects\Tiktoky`; the current Codex task cannot change its workspace root; cost if wrong: one final verified copy step is required before handoff.

## Preflight consistency scan

### Per-task internal checks

| Task | Tests versus implementation | Files versus later use | Finding or ruling |
| --- | --- | --- | --- |
| 1 | Strict config tests match Pydantic models and path resolution | Config and error types support all later tasks | `.gitignore` already exists for worktree isolation; Task 1 must extend it rather than recreate it. |
| 2 | Schema, adapter, and corruption tests match manifest contract | Manifest feeds Tasks 3, 4, 6, and 15 | Adapter fixtures and licence registry details are underspecified; implement the narrow behavior required by the spec and tests. |
| 3 | Balanced selector and shortcut audit tests match acquisition and audit behavior | Data policies feed Task 17 | Ruling: `minimum_generator_families` belongs at the primary-policy top level, not inside the WildFake source entry; this follows the prose and cross-source intent; cost if wrong: a WildFake-only gate would be stricter than intended. |
| 4 | Hash, near-duplicate, holdout, and leakage tests match split design | Split metadata feeds Tasks 9, 10, 15, and 17 | Ruling: add tests and implementation for `choose_holdout_generators`, which prose requires but the snippets omit; cost if wrong: automatic split selection could differ from an unstated manual choice. |
| 5 | Registry, determinism, severity, and family-mass tests match transforms | Shared by training, evaluation, inference, and web | Clean. |
| 6 | Dataset, sampling, and collation tests match paired-batch design | Feeds Tasks 7 through 9 | Stable seed and sampler concrete types are omitted from snippets; implement minimal typed contracts used by downstream code. |
| 7 | Output and freezing tests match DINOv2 wrapper | Used by losses, trainer, inference, and export | Ruling: add `preprocessing_version` to `Prediction` because the approved spec requires it; cost if wrong: callers must supply one additional metadata string. |
| 8 | Loss and margin tests match survival loss and mining | Trainer and E4 depend on it | `select_lowest_margin` is tested but omitted from code snippet; implement it as the public pure helper. |
| 9 | Checkpoint and tiny-training tests cover core persistence | Predictions and experiment execution depend on it | Ruling: persist Python, NumPy, and Torch RNG states and recover the source-manifest hash from split metadata; the spec requires both; cost if wrong: checkpoint files and metadata are slightly larger. |
| 10 | Metric and tie-break tests match validation-only ranking | Reporting, selection, and final test use it | `compute_condition_auc` is named in tests but only `_auc` appears in snippets; expose the public function and use it internally. |
| 11 | Calibration, threshold, table, plot, and gallery tests match reports | Inference, CLI, release, and experiments depend on artifacts | Ensure error categories use unique sample-condition rows and calibration metadata records the validation hash. |
| 12 | Probability and stability tests match service contract | ONNX, web, CLI, and acceptance depend on it | Add preprocessing version per Task 7 ruling and preserve CPU default behavior. |
| 13 | Export test covers tensor parity | Web and laptop acceptance depend on ONNX | Ruling: expand parity evidence to 32 samples and calibrated probability difference, because the prose and spec are stronger than the two-logit snippet; cost if wrong: integration test takes longer. |
| 14 | Upload tests match analysis result | CLI, docs, and acceptance depend on app | Add the required model-information accordion, limitations panel, disclaimer, parameters, and inference time omitted from the example layout. |
| 15 | CLI and miniature workflow tests match dispatch and artifact contract | Tasks 16 through 18 invoke these commands | Implement every thin handler referenced by `COMMAND_HANDLERS`; include export in the miniature workflow when ONNX dependencies are available. |
| 16 | Release tests match secret and document gates | Acceptance depends on it | Ruling: scan `git ls-files` in a repository and fall back to recursive files only for isolated test fixtures; the prose says tracked paths; cost if wrong: untracked local secrets are not a publication risk until staged, but pre-commit guidance must still warn users. |
| 17 | Commands and advancement rules match experiment design | Produces final artifacts for Task 18 | Full runs require licensed external datasets and likely external compute. Execute only after code gates pass; stop at this task only if those external prerequisites remain unavailable. |
| 18 | Laptop, clean-checkout, release, and judging checks match acceptance | Final handoff | A clean clone must use committed code and CPU ONNX; external publication remains human-only. |

### Shared-file and interface checks

| Producer task | Consumer task | Shared file or interface | Finding |
| --- | --- | --- | --- |
| 1 | 2, 3, 4, 6, 9, 10, 12, 13, 15 | Typed configuration and errors | Error hierarchy must remain stable. |
| 1 | 17 | `configs/experiments/e0_frozen.yaml` | Task 17 may modify only measured settings and must record changes. |
| 2 | 3 | Adapters, manifest schema, licence registry | Acquisition must emit adapter-compatible local paths and metadata. |
| 2 | 4 | Manifest DataFrame columns | Hashing and grouping may add columns but cannot rename canonical columns. |
| 2 | 6 | Manifest rows and image paths | Dataset must preserve IDs and generator metadata. |
| 2 | 15 | Manifest builder | CLI handlers must call package logic without duplication. |
| 3 | 15, 17 | Acquisition and data-policy configs | CLI must preserve manual WildFake guidance and top-level generator gate. |
| 4 | 9 | Split metadata and hashes | Trainer metadata must distinguish source manifest from split manifest. |
| 4 | 10 | Five partition names | Metrics must use validation partitions for selection and test partitions only after freeze. |
| 4 | 15, 17 | Split command and Parquet artifact | CLI must persist the policy and selected holdout families. |
| 5 | 6 | `TransformSpec`, registry, stable seed | Collator must apply transforms before DINO preprocessing. |
| 5 | 8 | Family-grouped registry | Hard miner must sample three distinct families uniformly. |
| 5 | 10 | Condition and family identifiers | Prediction records must use canonical IDs exactly. |
| 5 | 12, 14 | `apply_transform` and `get_spec` | UI stability view must use the same transform implementation as evaluation. |
| 6 | 7 | `[batch,3,224,224]` pixels | Model wrapper must not re-normalize processed tensors. |
| 6 | 8, 9 | `PairedBatch` and collator epoch | Trainer must set epoch before sampling transforms. |
| 7 | 8 | `DetectorOutput` and `LossBreakdown` | Shapes and normalized features must remain exact. |
| 7 | 9 | Trainable stages and forward output | E0 and E1 must differ only by stage and recorded configuration. |
| 7 | 12 | `Prediction` and Torch backend | Shared model and preprocessing versions must be returned. |
| 7 | 13 | Logit-only export wrapper | ONNX exports logits, while calibration remains outside the graph. |
| 8 | 9 | Survival loss and hard miner | Candidate discovery is no-gradient; chosen view is gradient-bearing. |
| 8 | 15, 17 | E3 and E4 configuration values | YAML values must be complete, explicit, and unchanged unless recorded. |
| 9 | 10 | Checkpoint IDs and prediction generation | Evaluation must load the selected atomic checkpoint and preserve run identity. |
| 9 | 15, 17 | `run_training` and run artifacts | CLI output layout must satisfy experiment commands. |
| 10 | 11 | `MetricReport` and prediction schema | Reports must preserve validation versus final-test provenance. |
| 10 | 15, 17 | Candidate selection and metrics JSON | Selection must never inspect test or generator-test results. |
| 11 | 12 | Calibration temperature and threshold | Inference uses temperature; decision reports use the validation-selected threshold. |
| 11 | 15, 16, 17 | Report artifacts | CLI, release docs, and experiment summaries use the same files. |
| 12 | 13 | `LogitBackend` | ONNX image backend must satisfy the same protocol as Torch. |
| 12 | 14 | `InferenceService` and stability result | Web layer contains no duplicate model logic. |
| 12 | 15, 18 | CPU service construction | CLI must default to CPU-safe execution. |
| 13 | 14, 15, 18 | ONNX model and backend | Final app must run without CUDA and use verified calibration metadata. |
| 13 | 16 | ONNX CI parity | Optional OpenVINO may skip, but ONNX CPU may not. |
| 14 | 15, 16, 18 | Gradio app factory and launch command | Manual smoke check must use a real exported or fixture model. |
| 15 | 16 | Documented CLI and miniature reproduction | README commands must be verified against parser behavior. |
| 15 | 17 | Experiment commands and output layout | Every Task 17 command must parse and dispatch successfully. |
| 15 | 18 | Clean-checkout reproduction | Miniature workflow cannot require data or network. |
| 16 | 18 | README, model card, Devpost draft, video script | Task 18 updates measured results without removing required sections. |
| 17 | 18 | Selection, calibration, ONNX, and final reports | Acceptance may begin only after model selection is frozen.

Preflight status: ready to dispatch Task 1 with the rulings above.

Baseline: branch `codex/prooflens-implementation` at `9b16823`; repository contains no package or tests yet, so baseline tests are not applicable; bundled Python 3.12.13 selected because system Python 3.14.2 violates project requirements.

Task 1: dispatched implementer `/root/task_1_implementer`; base `9b16823`.
Task 1: implementation commit `0d7693c`; focused tests 2 passed, full suite 2 passed, Ruff passed; reported pytest cache warning; review dispatched to `/root/task_1_reviewer`.
Task 1: review needs fixes: non-clean pytest output, invalid RED evidence, and missing MIT declaration.
Task 1: Ruling: add SPDX MIT metadata in Task 1, while the full `LICENSE` text and third-party notices remain in Task 16 because the locked file map assigns them there; cost if wrong: licence text is unavailable until Task 16, but package intent is machine-readable immediately.
Task 1: fix round 1/5 (3 addressed, 0 open; commits `0d7693c..d00cd08`).
Task 1: complete (commits `9b16823..d00cd08`, review clean).
Task 2: dispatched implementer `/root/task_2_implementer`; base `d00cd08`.
Task 2: implementation commit `0696533`; focused tests 6 passed, full suite 8 passed, Ruff passed; review dispatched to `/root/task_2_reviewer`.
Task 2: review requested fixes: SID root scanning can mislabel excluded class 2, adapters do not fail on missing structure, Parquet atomicity lacks direct tests, and full zero-row schema coverage is missing.
Task 2: fix round 1/5 dispatched to original implementer `/root/task_2_implementer`.
Task 2: fix round 1/5 produced commits `6518426..1424c23`; RED 3 failed and 9 passed, GREEN 14 passed, full suite 16 passed, Ruff passed; fresh re-review dispatched to `/root/task_2_rereviewer`.
Task 2: complete (commits `d00cd08..1424c23`, review clean).
Task 3: brief generated; base `1424c23`.
Task 3: dispatched implementer `/root/task_3_implementer`; base `1424c23`.
Task 3: implementation commits `c3db09c..f5e016e`; focused tests 15 passed, full suite 31 passed, Ruff passed.
Workspace: migrated the clean implementation snapshot into the GitHub-backed delivery repository on branch `codex/prooflens-implementation` at commit `6b9f6e2`; future work uses `C:\Users\Loh Yang Zhi\Documents\Projects\Tiktoky`.
Task 3: independent review dispatched to `/root/task_3_reviewer`.
Task 3: review requested fixes for safe acquisition paths, missing or blank dataset names, production WildFake guidance, and deterministic tied-count rendering.
Task 3: fix round 1/5 dispatched to original implementer `/root/task_3_implementer` in the delivery repository.
Task 3: fix round 1/5 produced commits `e8946e1..a26a50b`; RED 11 failed, GREEN 37 passed, full suite 42 passed, Ruff passed; fresh re-review dispatched to `/root/task_3_rereviewer`.
Task 3: re-review requested fix round 2 for non-string or null path values and collisions among images, manifest, and reserved acquisition metadata destinations.
Task 3: fix round 2/5 dispatched to original implementer `/root/task_3_implementer`.

## Execution protocol update

User-approved milestone review mode replaces per-task independent review from this point forward. Each task still uses test-driven development, focused verification, and local recovery commits. Independent review, fix consolidation, full-suite verification, and the reviewable Git handoff occur once per milestone. No remote push is authorized.

- Milestone 1: Tasks 1-6, complete data pipeline.
- Milestone 2: Tasks 7-9, model and training core.
- Milestone 3: Tasks 10-13, evaluation through ONNX inference.
- Milestone 4: Tasks 14-16, app, CLI, documentation, and release gates.
- Milestone 5: Tasks 17-18, experiments, laptop acceptance, and final handoff.

Milestone 1: in progress; Task 3 fix round 2 resumed, followed by Tasks 4-6 before one consolidated review.
Task 3: fix round 2 complete in commits `13f81f8..2f91fac`; RED 23 failed, edge-case GREEN 23 passed, focused acquisition suite 43 passed, Ruff passed. Consolidated re-review deferred to Milestone 1 gate by user-approved protocol.
Task 4: dispatched implementer `/root/task_4_implementer`; base `2f91fac`.
Task 4: original implementer interrupted after creating untracked hashing, splitting, and focused test files; no commit created.
Task 4: user requested higher reasoning; partial work handed to fresh ultra-reasoning implementer `/root/task_4_ultra_implementer` with the locked Task 4 requirements unchanged.
Task 4: complete in commits `53dae17..0f0bbd8`; behavioral RED 16 failed and 44 passed plus provenance RED 2 failed, final focused suite 61 passed, Ruff passed. Full-suite and independent review deferred to Milestone 1 gate.
Task 5: dispatched implementer `/root/task_5_implementer`; base `0f0bbd8`.
