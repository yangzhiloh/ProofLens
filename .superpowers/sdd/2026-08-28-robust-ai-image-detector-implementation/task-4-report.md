# Task 4 implementation report

## Result

Task 4 is implemented in commit `53dae17521255b1373bf17e5ca6516733b63fd4d`.

## Files

1. `src/prooflens/data/hashing.py`
2. `src/prooflens/data/splitting.py`
3. `tests/unit/data/test_hashing.py`
4. `tests/unit/data/test_splitting.py`

## Behavioral decisions

1. SHA-256 hashes exact file bytes in bounded chunks. Perceptual hashing applies EXIF orientation, converts to RGB, and returns the 64-bit `imagehash.phash` hexadecimal form. Missing, blank, non-path, unreadable, and undecodable inputs raise `DataIntegrityError`. Hash enrichment deep-copies its input.
2. Duplicate grouping validates unique nonblank sample identifiers, nonblank source identifiers, 64-character SHA-256 values, and 16-character pHash values. SHA and pHash text is case-normalized, while source and sample identities remain case-sensitive.
3. Source identity, exact checksum identity, and every pHash neighbor within the inclusive Hamming radius are unioned transitively. pHash radius lookup uses an exact BK-tree. A component identifier is the first 16 hexadecimal characters of SHA-256 over the component's lexicographically smallest sample ID.
4. Automatic generator holdouts use fake-row counts only. Descending count and ascending lexical family name determine rank. The largest family is generator test and the second is generator validation. Production requires 100 rows per chosen family and at least three fake families. `minimum_holdout_family_rows` models an explicit lower fixture threshold with production default 100.
5. Manual holdouts must be nonempty, disjoint, present among fake rows, and meet the configured threshold. A connected component spanning both holdouts raises `LeakageError`.
6. The splitter creates exactly `train`, `validation`, `test`, `generator_validation`, and `generator_test`. Entire connected components follow generator holdouts. Disjoint authentic-only groups supply both generator partitions. Remaining groups use deterministic 15 percent validation and test allocation with at most 100 seeded retries, with typed class and group diagnostics when both labels cannot be preserved.
7. Leakage validation rejects null or blank checksum, source, and split-group identities. It allows repeats within one partition and detects checksum leakage independent of hexadecimal letter case.
8. Split persistence requires a `.parquet` destination. It sorts rows stably, writes Parquet plus adjacent canonical JSON through temporary files, records policy, resolved holdouts, row counts, group counts, exact source-manifest SHA-256, and SHA-256 of the persisted split Parquet bytes. State-aware backup rollback preserves both prior outputs even when the second backup operation fails.

## TDD evidence

### Inherited baseline

With this repository's `src` directory selected explicitly, the interrupted four-file implementation collected 29 tests and passed all 29. That result was not accepted as RED evidence. Earlier collection and fixture errors were traced to a stale editable install pointing at the retired worktree and sandbox-denied default temporary storage, so they were excluded as environmental rather than behavioral failures.

Baseline command:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src')
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/unit/data/test_hashing.py tests/unit/data/test_splitting.py -v --basetemp '.pytest-task4-baseline'
```

Evidence: `29 passed in 1.45s`.

### RED cycle 1

The repaired suite replaced the incorrect null-identity allowance and added missing behavior for blank paths, public fixture thresholds, typed policy preflight, BK-tree oracle agreement, connected holdout rejection, authentic-group disjointness, input non-mutation, Parquet destination validation, metadata policy, and rollback after metadata backup failure.

Command:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src')
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/unit/data/test_hashing.py tests/unit/data/test_splitting.py -v --basetemp '.pytest-task4-red'
```

Evidence: `16 failed, 44 passed in 1.63s`. Failures were the expected blank-path handling, missing public threshold API and policy field, blank manual family validation, raw seed and fraction failures, null or blank leakage identities, missing threshold metadata, non-Parquet destination acceptance, and deletion of prior metadata during backup failure.

The first GREEN attempt produced `1 failed, 59 passed`. The one failure exposed an invalid test expectation because the fixture made `gamma` the largest family. The fixture was corrected to literal counts `alpha=20`, `beta=19`, and `gamma=18` without changing production code. The next focused run produced `60 passed in 1.33s`.

### RED cycle 2

Refactor review identified two further provenance behaviors and added focused tests.

Command:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src')
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/unit/data/test_splitting.py::test_assert_no_leakage_normalizes_sha256_hex_case tests/unit/data/test_splitting.py::test_split_writer_round_trips_deterministic_parquet_and_metadata -v --basetemp 'C:\Users\Loh Yang Zhi\Documents\ChatGPT\Tiktoky\.pytest-task4-red2'
```

Evidence: `2 failed in 0.72s`. Checksum case variants crossed partitions undetected, and `split_sha256` did not equal SHA-256 of the persisted Parquet bytes.

After the minimal implementation, the same two tests produced `2 passed in 0.45s`.

## Final verification

Focused Task 4 verification:

```powershell
$env:PYTHONPATH = (Resolve-Path 'src')
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/unit/data/test_hashing.py tests/unit/data/test_splitting.py -v --basetemp 'C:\Users\Loh Yang Zhi\Documents\ChatGPT\Tiktoky\.pytest-task4-verify'
```

Evidence: `61 passed in 1.75s`, exit code 0.

Changed-file Ruff verification:

```powershell
& 'C:\Users\Loh Yang Zhi\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m ruff check --no-cache src/prooflens/data/hashing.py src/prooflens/data/splitting.py tests/unit/data/test_hashing.py tests/unit/data/test_splitting.py
```

Evidence: `All checks passed!`, exit code 0.

## Deferred gates and concerns

1. Per the user-approved milestone protocol, the full repository suite and independent review are deferred to the Milestone 1 gate after Task 6.
2. The runtime's editable installation still points at the retired worktree. Verification therefore set `PYTHONPATH` to this delivery repository's `src` directory. This affects the local test invocation, not committed package behavior.
3. A BK-tree is an exact metric index and avoids an explicit all-pairs scan. As with metric trees generally, adversarial data can degrade its practical traversal, but the implementation satisfies the required genuine radius index and matches the brute-force oracle on all tested radii.
4. No remote push, merge, or amendment was performed.
