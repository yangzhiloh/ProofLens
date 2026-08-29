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

