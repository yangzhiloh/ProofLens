# Task 6 implementation report

## Result

Task 6 is implemented in commit `67979b182dd511e39f47d9961aa54c50f4daacc2`.

## Files

1. `src/prooflens/data/dataset.py`
2. `src/prooflens/data/sampling.py`
3. `src/prooflens/data/collate.py`
4. `src/prooflens/inference/__init__.py`
5. `src/prooflens/inference/preprocess.py`
6. `tests/unit/data/test_dataset.py`
7. `tests/unit/data/test_sampling.py`
8. `tests/unit/data/test_collate.py`
9. `tests/unit/inference/test_preprocess.py`

## Behavioral decisions

1. `SourceImageDataset` snapshots assigned canonical rows positionally, rather than using pandas index labels. It requires every canonical manifest column plus `split_group_id`, rejects duplicate or blank sample IDs, validates binary integer labels and the five assigned split names, and never mutates the caller's frame.
2. `SourceItem` is a frozen, slotted dataclass containing the detached RGB image, label, sample and source identity, dataset and generator identity, split, and split-group metadata.
3. Source decoding performs a Pillow verification pass, reopens the file, applies EXIF orientation, forces pixel loading, converts to RGB, copies the pixels, and closes both file contexts before returning. Missing, corrupt, truncated, directory, and unsupported inputs become `ImageDecodeError` values containing the sample ID and path.
4. `compute_sampling_weights()` uses positional NumPy output. Authentic strata are `(label, dataset_name)` and fake strata are `(label, dataset_name, generator_family)`. Each label receives mass `0.5`, its strata divide that mass equally, and rows divide stratum mass equally. Both labels and all stratum fields used by a row are validated before weights are computed.
5. `make_weighted_sampler()` owns an explicit `torch.Generator`, validates the seed and draw count, and passes the generator directly to `WeightedRandomSampler`. It does not consume Torch, NumPy, or Python global random state.
6. `stable_seed()` uses SHA-256 over a versioned domain plus typed, length-delimited primitive components. `None`, booleans, integers, finite floats, strings, and bytes have distinct tags. The result is a nonnegative 63-bit value accepted by Torch and NumPy. Python `hash()` is never used.
7. `TransformSampler` is an injectable protocol. `FixedTransformSampler` resolves one Task 5 condition. `FamilyBalancedTransformSampler` uses a local PRNG seeded from run seed, epoch, sample ID, and a domain component, then selects a canonical family uniformly and a severity uniformly within that family.
8. `PairedBatchCollator` rejects empty batches, non-`SourceItem` values, blank sample IDs, invalid labels, and invalid epochs. It accepts repeated sample IDs because the weighted training sampler draws with replacement. It derives one transform pixel seed from run seed, epoch, and stable sample ID, applies Task 5 transformations before preprocessing, and copies source images before passing them to the injected processor.
9. `PairedBatch` is a frozen, slotted dataclass. It contains clean and transformed float tensors, float labels, condition and sample IDs, and dataset, generator, source-group, split, and split-group tuples. The collator constructs every metadata tuple at the batch length.
10. `preprocess_images()` is the single processor seam used by collation and later inference. It calls the injected processor with `return_tensors="pt"`, performs no independent resize or DINO normalization, converts valid floating results to `float32`, and rejects missing, non-tensor, wrong-rank, wrong-batch, wrong-shape, non-floating, or nonfinite outputs with `DataIntegrityError`.
11. `create_dinov2_processor()` imports Transformers only inside the explicit production factory and calls `AutoImageProcessor.from_pretrained("facebook/dinov2-base")`. The shared metadata identifier is `dinov2-base-224-v1`. Tests replace the processor and import boundary with network-free fakes.

## TDD evidence

### RED

All Task 6 behavioral tests were created before any of the Task 6 production modules existed. Imports occur inside test bodies so the missing feature produces valid test failures rather than collection errors.

Command:

```powershell
$env:PYTHONPATH='src'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/unit/data/test_dataset.py tests/unit/data/test_sampling.py tests/unit/data/test_collate.py tests/unit/inference/test_preprocess.py -p no:cacheprovider --basetemp='tests/.pytest-task6-red' -q
```

Exact result: `53 failed, 3 warnings in 4.60s`, exit code 1. Every failure was `ModuleNotFoundError` for a missing Task 6 module inside a behavioral test. The three pandas assignment warnings were removed before final verification without changing expectations.

### First GREEN

The first implementation run required no behavioral correction.

Exact result: `53 passed in 10.63s`, exit code 0.

### Final GREEN

Command:

```powershell
$env:PYTHONPATH='src'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/unit/data/test_dataset.py tests/unit/data/test_sampling.py tests/unit/data/test_collate.py tests/unit/inference/test_preprocess.py -p no:cacheprovider --basetemp='tests/.pytest-task6-green-final' -q
```

Exact result: `53 passed in 10.80s`, exit code 0, with no warnings.

## Ruff evidence

The first changed-file Ruff run found two mechanical diagnostics: one unused `Sequence` import and one import-style diagnostic in the sampling test helper. Both were corrected without changing behavior.

Final command:

```powershell
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ruff check --no-cache src/prooflens/data/dataset.py src/prooflens/data/sampling.py src/prooflens/data/collate.py src/prooflens/inference/__init__.py src/prooflens/inference/preprocess.py tests/unit/data/test_dataset.py tests/unit/data/test_sampling.py tests/unit/data/test_collate.py tests/unit/inference/test_preprocess.py
```

Exact result: `All checks passed!`, exit code 0.

Staged diff verification before the implementation commit:

```powershell
git diff --cached --check
```

Evidence: no output, exit code 0.

## Coverage

The original 53 focused cases cover canonical assigned-frame validation, duplicate and non-default pandas indices, positional order, caller nonmutation, frozen source items, pixel-derived dimensions, RGB conversion, EXIF orientation, missing, corrupt, truncated, and unsupported decode failures, eager file closure, exact label and stratum mass, missing strata, invalid and single labels, finite positive normalized weights, deterministic weighted sampling, global RNG isolation, typed stable-seed vectors across processes and `PYTHONHASHSEED` values, component ambiguity separation, primitive validation, fixed and family-balanced transform selection, family then severity distribution, paired tensor shapes and dtypes, labels and manifest metadata, epoch determinism, batch-order independence, transformed-view difference, processor reuse, source nonmutation even with a mutating fake, empty batches, invalid epochs, every malformed processor-output category, injected shared preprocessing, preprocessing version metadata, and lazy no-network processor construction. The Milestone 1 review added one integration regression covering duplicate IDs produced by the replacement sampler.

## Milestone 1 review amendment

The consolidated local review found that `make_weighted_sampler()` defaults to replacement sampling while `PairedBatchCollator` rejected duplicate sample IDs. A two-row, four-draw regression test failed at the uniqueness guard before the production change. Removing only that guard made the regression pass, and the complete Task 6 suite then passed 54 tests. The final Milestone 1 gate passed 219 tests in 12.20 seconds, and Ruff reported no diagnostics.

## Deferred gates and concerns

1. Per the user-approved milestone protocol, the full repository suite and independent review are deferred to the Milestone 1 gate immediately after this task.
2. The editable installation still points at the retired worktree. Every pytest command used bundled Python 3.12.13 with explicit `PYTHONPATH=src`, bytecode and pytest plugin autoload disabled, cache disabled, and a checkout-local basetemp. All basetemp directories created by this task were removed after verification.
3. The source dataset intentionally requires the complete canonical assigned manifest, including `split_group_id`. Callers with pre-split or partial frames receive a typed integrity error instead of implicit defaults.
4. The production Transformers factory was verified through a fake import and exact model-ID assertion. It was not invoked against the network and no model weights were downloaded.
5. Processor outputs are required to be floating tensors before conversion to `float32`. Integer processor outputs are rejected because converting raw integer pixels without the processor's normalization would silently create an invalid DINO input.
6. No full suite, remote push, merge, or amendment was performed.
