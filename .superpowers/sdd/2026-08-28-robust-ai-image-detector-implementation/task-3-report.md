# Task 3 report: dataset acquisition and source-shortcut audit

## Outcome

Task 3 implements offline-testable SID-Set streaming acquisition, manual WildFake root guidance, primary manifest policy gates, CIFAKE stress-only policy, and JSON plus Markdown source-shortcut audits. Task 2 manifest APIs and schema remain unchanged.

## Files

1. `src/prooflens/data/acquire.py`
2. `src/prooflens/data/audit.py`
3. `src/prooflens/errors.py`
4. `configs/data/sid_subset.yaml`
5. `configs/data/wildfake.yaml`
6. `configs/data/cifake.yaml`
7. `configs/data/primary.yaml`
8. `tests/unit/data/test_acquire.py`
9. `tests/unit/data/test_audit.py`

## Decisions

1. SID-Set acquisition calls the injected or Hugging Face loader for `saberzl/SID_Set` with `split="train"`, `streaming=True`, and an explicit pinned revision. The checked-in policy pins `c1674903d858c78e04809c1c6f2703627ac1a621` and defaults to 10,000 rows per binary class.
2. Selection excludes label `2`, stops immediately after exactly reaching both class caps, and rejects underfilled streams with `DatasetAcquisitionError`.
3. Acquisition saves decoded source images as RGB PNG files in a staging directory. It preserves `img_id` in `sample_id`, `original_image_id`, and source grouping fields, then builds and validates Parquet through Task 2's `SidSetAdapter`, `ManifestRecord`, and `build_manifest` contracts. Publication relocates paths and atomically replaces the canonical manifest.
4. Existing acquisition roots are never overwritten. A failed or underfilled acquisition removes its private staging directory and publishes no partial output.
5. `acquisition.json` records the configured revision, loader-observed revision when exposed, CC-BY-4.0, selected counts, split, dataset identifier, and SHA-256 of canonical sorted configuration JSON.
6. WildFake remains a manual acquisition. Missing or empty roots identify both the official repository and ModelScope dataset page. The adapter remains responsible for validating the configured export hierarchy.
7. The primary policy places `minimum_generator_families: 3` at the top level. Validation requires both labels, source-specific allowed labels, approved sources only, and three distinct fake generator families across every approved source marked `generator_labeled`.
8. CIFAKE is explicitly `stress_only: true`, excluded from primary training, and absent from `primary.yaml`.
9. Audit output includes row, class, dataset, and generator counts; width and height quantiles; file-format by-label cross-tabs; missing metadata counts including blank strings; exact duplicate count; and meaningful categorical perfect shortcuts.
10. Perfect-shortcut detection ignores missing and blank values, requires at least two observed categories and both observed labels, and only flags a feature when each observed category maps to exactly one label. Empty and all-missing columns are safe.
11. If `content_checksum` is absent, the audit explicitly records every row as missing for that field and reports zero exact duplicates. Blank checksums receive the same missing treatment and do not create false duplicate counts.
12. Audit JSON, audit Markdown, acquisition metadata, and canonical manifest replacement use sibling temporary files followed by atomic replacement where overwrite risk exists.

## TDD evidence

### Initial RED

Command:

```powershell
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/unit/data/test_acquire.py tests/unit/data/test_audit.py -v --basetemp .pytest-tmp-task3-red
```

Exact result: collection stopped with `2 errors in 0.55s`. `prooflens.data.acquire` and `prooflens.data.audit` were both missing. This was the expected RED caused by absent Task 3 behavior, not by pytest or dependency failures.

### First GREEN attempt and correction

Command:

```powershell
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/unit/data/test_acquire.py tests/unit/data/test_audit.py -v --basetemp .pytest-tmp-task3-green
```

Exact result: `1 failed, 13 passed in 1.16s`. The persisted manifest retained staging-directory image paths after publication. The implementation was corrected to rebuild the final manifest atomically against relocated paths.

Follow-up command:

```powershell
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/unit/data/test_acquire.py tests/unit/data/test_audit.py -v --basetemp .pytest-tmp-task3-green2
```

Exact result: `14 passed in 0.62s`.

### Pinned-policy RED and GREEN

Command:

```powershell
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/unit/data/test_acquire.py::test_dataset_configs_pin_sid_defaults_and_keep_cifake_stress_only -v --basetemp .pytest-tmp-task3-red-policy
```

Exact RED result: `1 failed in 0.65s`. The draft policy used mutable revision `main`, whose length was 4 rather than a full 40-character pinned revision.

Follow-up command:

```powershell
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/unit/data/test_acquire.py::test_dataset_configs_pin_sid_defaults_and_keep_cifake_stress_only -v --basetemp .pytest-tmp-task3-green-policy
```

Exact GREEN result: `1 passed in 0.41s`.

## Final verification evidence

### Focused tests

Command:

```powershell
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/unit/data/test_acquire.py tests/unit/data/test_audit.py -v --basetemp .pytest-tmp-task3-resume-focused
```

Exact result: `15 passed in 0.54s`.

### Full suite

Command:

```powershell
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest -v --basetemp .pytest-tmp-task3-resume-full
```

Exact result: `31 passed in 0.77s`.

### Ruff

Command:

```powershell
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ruff check src tests
```

Exact result: `All checks passed!`

## Commits

1. `c3db09cbcf5c75db18dcc1f04f7b1b42eca53c68` `feat: add reproducible data acquisition and audits`
2. `78d48f6f8a1f799adf460f6971deae0b7d297fe6` `style: remove trailing audit blank lines`

No prior commit was amended.

## Concerns

1. The real 20,000-image SID-Set acquisition was not run because Task 3 tests must remain network-free and the source dataset is large. The injected streaming loader exercises acquisition, RGB persistence, canonical manifest construction, metadata, and underfill behavior with tiny fixtures.
2. WildFake redistribution rights still require human verification before acquisition or publication, as recorded by `REQUIRES-VERIFICATION`.
3. Some Hugging Face iterable dataset objects do not expose an observed repository revision. In that case, `observed_dataset_revision` deliberately falls back to the configured pinned revision rather than inventing a value.
