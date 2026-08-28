# Task 2 Report: Canonical manifest schema and dataset adapters

## Implementation summary

Added the strict `ManifestRecord` schema and stable DataFrame serialization, a verified licence and attribution registry, SID-Set, WildFake, and CIFAKE local adapters, and atomic Parquet manifest construction. The builder decodes images using EXIF transpose plus RGB conversion, overwrites manifest dimensions and format with decoded metadata, records corrupt paths, and raises `ManifestBuildError` when the configured corrupt fraction is exceeded. CIFAKE rows use `dataset_name="cifake_stress"`; SID label `2` is omitted; WildFake fake generator identity comes from the configured fake-directory hierarchy.

## Files changed

- `src/prooflens/data/__init__.py`
- `src/prooflens/data/schema.py`
- `src/prooflens/data/licences.py`
- `src/prooflens/data/manifest.py`
- `src/prooflens/data/adapters/__init__.py`
- `src/prooflens/data/adapters/base.py`
- `src/prooflens/data/adapters/sid_set.py`
- `src/prooflens/data/adapters/wildfake.py`
- `src/prooflens/data/adapters/cifake.py`
- `tests/unit/data/test_schema.py`
- `tests/unit/data/test_adapters.py`

## RED evidence

Command:

```powershell
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/unit/data/test_schema.py tests/unit/data/test_adapters.py -v
```

Exact result:

```text
collecting ... collected 0 items / 2 errors
tests/unit/data/test_schema.py:5: in <module>
    from prooflens.data.schema import ManifestRecord, records_to_frame
E   ModuleNotFoundError: No module named 'prooflens.data'
tests/unit/data/test_adapters.py:7: in <module>
    from prooflens.data.adapters.cifake import CifakeAdapter
E   ModuleNotFoundError: No module named 'prooflens.data'
============================== 2 errors in 0.18s ==============================
```

## GREEN evidence

Command:

```powershell
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/unit/data/test_schema.py tests/unit/data/test_adapters.py -v --basetemp .pytest-tmp
```

Exact result:

```text
collected 6 items
tests/unit/data/test_schema.py::test_manifest_rejects_non_binary_primary_label PASSED
tests/unit/data/test_schema.py::test_records_to_frame_has_stable_column_order PASSED
tests/unit/data/test_adapters.py::test_sid_adapter_excludes_tampered_label_two PASSED
tests/unit/data/test_adapters.py::test_wildfake_adapter_reads_generator_from_hierarchy PASSED
tests/unit/data/test_adapters.py::test_cifake_adapter_marks_records_as_stress_only PASSED
tests/unit/data/test_adapters.py::test_manifest_builder_stops_above_corrupt_limit PASSED
============================== 6 passed in 0.96s ==============================
```

The workspace-local `--basetemp` is required because the host's default pytest Temp root returns `PermissionError: [WinError 5] Access is denied`.

## Full-suite and Ruff evidence

Full suite command:

```powershell
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest -v --basetemp .pytest-tmp-full
```

Output: `8 passed in 0.70s`.

Ruff command:

```powershell
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ruff check src tests
```

Output: `All checks passed!`

## Self-review

- `ManifestRecord` forbids extra keys, accepts only labels 0 and 1, and retains declaration-order columns for empty and populated frames.
- Adapter scans are deterministic through sorted local paths; WildFake's directory to generator mapping is explicit and configurable.
- The builder only replaces the destination after Parquet serialization succeeds, removes an unpromoted temporary file, and counts every undecodable record before threshold evaluation.
- Existing `.gitignore` intentionally ignores any directory called `data`; the explicitly requested source and test files must therefore be force-added by path. No ignore policy is changed.

## Concerns

- The host default pytest temporary directory is inaccessible, so verification commands use workspace-local `--basetemp` paths.

## Review fix round 1

### Changes

- SID-Set `scan()` now rejects root-only scanning with `DataIntegrityError`; `scan_rows()` is the only path that accepts verified source labels and still omits label `2`.
- WildFake and CIFAKE validate the configured root, each required mapped directory, and the presence of images for both classes before producing records.
- Persistence tests now verify full Parquet schema readback, decoded dimensions and format, temporary-file cleanup, atomic replacement, and destination preservation after a forced serialization failure.
- `records_to_frame([])` retains every canonical schema column and `.gitignore` now root-anchors raw `/data/` so source and test data modules are not ignored.

### RED evidence

Command:

```powershell
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/unit/data/test_schema.py tests/unit/data/test_adapters.py -v --basetemp .pytest-tmp-red-round1
```

Exact result: `3 failed, 9 passed in 2.63s`. The failed behaviors were: SID root scanning did not raise metadata validation, WildFake accepted a missing `fake` directory, and CIFAKE accepted a missing `FAKE` directory.

### GREEN evidence

Command:

```powershell
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/unit/data/test_schema.py tests/unit/data/test_adapters.py -v --basetemp .pytest-tmp-green-round1
```

Exact result: `14 passed in 0.57s`.

### Full suite and Ruff

```powershell
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest -v --basetemp .pytest-tmp-full-round1
```

Exact result: `16 passed in 0.53s`.

```powershell
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ruff check src tests
```

Exact result: `All checks passed!`

### Commit

`65184263b65016550d5bae6be11b94b394b64c1f` `fix: validate dataset adapter roots`
