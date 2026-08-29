# Dataset policy and acquisition

ProofLens keeps raw data outside Git and records dataset identity, version, licence identifier,
labels, and generator metadata in the canonical manifest. A source is not eligible for primary
training until its structure, licence, class balance, formats, resolutions, content, and
generator coverage have been audited.

## Source registry

| Source | Role | Local root | Labels | Version or revision | Licence status |
| --- | --- | --- | --- | --- | --- |
| SID-Set | Primary candidate | `data/raw/sid_set` | `0` authentic, `1` AI-generated | Pinned revision `c1674903d858c78e04809c1c6f2703627ac1a621` | CC-BY-4.0; source-material attribution required |
| WildFake | Primary candidate and generator-family source | `data/raw/wildfake` | `0` authentic, `1` AI-generated | Local export recorded as configured | REQUIRES-VERIFICATION before redistribution |
| CIFAKE | Low-resolution stress test only | `data/raw/cifake` | `REAL` becomes `0`, `FAKE` becomes `1` | Local export recorded as configured | MIT; retain required citations |

Primary policy requires both labels and at least three generator families from approved
generator-labelled sources. CIFAKE cannot enter primary training under the current policy.

## SID-Set automated acquisition

`configs/data/sid_subset.yaml` identifies `saberzl/SID_Set`, streams its `train` split at the
pinned revision, and selects 10,000 rows per binary class. Label values outside `0` and `1` are
excluded. The operation fails without publishing a partial destination if either class is
underfilled.

```text
python -m prooflens.cli acquire --config configs/data/sid_subset.yaml --output data/raw/sid_set
```

The destination must not already exist. A successful acquisition contains RGB PNG files,
`manifest.parquet`, and `acquisition.json`. The metadata records requested and observed revision,
class counts, licence identifier, and a configuration hash. Network access is required. If the
provider requires authentication or acceptance of terms, complete that human-controlled step
before running the command.

## WildFake manual placement

Acquire WildFake only after reviewing the terms at the official repository and ModelScope source
recorded in `configs/data/wildfake.yaml` and `THIRD_PARTY_NOTICES.md`. Extract it to:

```text
data/raw/wildfake/
    real/
        any-nested-authentic-files
    fake/
        generator-family-a/
            any-nested-generated-files
        generator-family-b/
            any-nested-generated-files
        generator-family-c/
            any-nested-generated-files
```

The adapter requires nonempty `real/` and `fake/` branches. Each direct child of `fake/` is used
as a generator-family identifier. It does not infer uncertain labels. The repository marks the
licence as REQUIRES-VERIFICATION and does not automate redistribution.

## CIFAKE stress placement

Place an approved CIFAKE export at:

```text
data/raw/cifake/
    REAL/
    FAKE/
```

Both directories must be nonempty. The adapter records generated CIFAKE rows with the configured
generator family and keeps the dataset separate from primary training.

## Canonical manifest, audit, and split

```text
python -m prooflens.cli manifest --config configs/data/primary.yaml --output artifacts/manifests/primary.parquet
python -m prooflens.cli audit --manifest artifacts/manifests/primary.parquet --output artifacts/reports/data-audit
python -m prooflens.cli split --manifest artifacts/manifests/primary.parquet --output artifacts/manifests/primary-split.parquet --seed 17
```

Each canonical row records sample and source-group identifiers, path, binary label, dataset and
version, generator family, original image identifier, dimensions, format, licence identifier,
content checksum, perceptual hash, and split. The builder applies EXIF orientation, decodes to
RGB, logs corrupt images, and fails when corruption exceeds the configured fraction.

The audit reports class, dataset, generator, resolution, format, metadata, duplicate, and
label-predictive categorical distributions. Splitting keeps source groups and duplicate clusters
together, preserves both labels, and reserves generator-family partitions. Exact checksum or
source-group leakage across assigned partitions is a hard failure.

## Publication boundary

`/data/` and `/artifacts/` are ignored by Git. `scripts/release_check.py` inspects only
`git ls-files` in a repository and rejects any tracked `data/raw/` file. This prevents local,
ignored dataset copies from becoming false release failures while still blocking accidental
publication.
