from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from prooflens.data.splitting import (
    PARTITIONS,
    HoldoutSelection,
    SplitPolicy,
    assert_no_leakage,
    assign_grouped_splits,
    build_split_groups,
    choose_holdout_generators,
    split_metadata_path,
    write_split_manifest,
)
from prooflens.errors import DataIntegrityError, LeakageError


def _checksum(value: str) -> str:
    return hashlib.sha256(f"checksum:{value}".encode()).hexdigest()


def _phash(value: str) -> str:
    return hashlib.sha256(f"phash:{value}".encode()).hexdigest()[:16]


def _brute_phash_connections(values: list[str], radius: int) -> set[tuple[int, int]]:
    parent = list(range(len(values)))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(first: int, second: int) -> None:
        left = find(first)
        right = find(second)
        if left != right:
            parent[right] = left

    integers = [int(value, 16) for value in values]
    for first in range(len(integers)):
        for second in range(first):
            if (integers[first] ^ integers[second]).bit_count() <= radius:
                union(first, second)
    return {
        (first, second)
        for first in range(len(values))
        for second in range(first + 1, len(values))
        if find(first) == find(second)
    }


def _row(
    sample_id: str,
    *,
    label: int = 0,
    family: str = "authentic",
    source_group: str | None = None,
    checksum: str | None = None,
    phash: str | None = None,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "label": label,
        "generator_family": family,
        "source_group_id": source_group or f"source:{sample_id}",
        "content_checksum": checksum or _checksum(sample_id),
        "perceptual_hash": phash or _phash(sample_id),
    }


def _holdout_counts(
    counts: dict[str, int], *, real_groups: int = 0
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family, count in counts.items():
        rows.extend(
            _row(f"{family}:{index:03d}", label=1, family=family)
            for index in range(count)
        )
    rows.extend(_row(f"real:{index:03d}") for index in range(real_groups))
    return pd.DataFrame(rows)


def _assignable_frame() -> pd.DataFrame:
    # Explicit row-count fixture: no image payloads, 100 rows per fake family.
    return _holdout_counts(
        {"alpha": 100, "beta": 100, "gamma": 100},
        real_groups=60,
    )


def _automatic_policy(seed: int = 17) -> SplitPolicy:
    return SplitPolicy(
        seed=seed,
        validation_fraction=0.15,
        test_fraction=0.15,
        generator_validation_families=frozenset(),
        generator_test_families=frozenset(),
        generator_evaluation_real_fraction=0.10,
    )


def test_choose_holdouts_uses_row_counts_and_lexical_tie_break() -> None:
    frame = _holdout_counts({"zeta": 101, "beta": 120, "alpha": 120})

    selection = choose_holdout_generators(frame)

    assert selection == HoldoutSelection(
        generator_validation_families=frozenset({"beta"}),
        generator_test_families=frozenset({"alpha"}),
    )


def test_choose_holdouts_requires_three_total_families_and_two_eligible() -> None:
    with pytest.raises(DataIntegrityError, match="at least three.*found 2"):
        choose_holdout_generators(_holdout_counts({"alpha": 100, "beta": 100}))

    with pytest.raises(DataIntegrityError, match="at least 100.*gamma=99"):
        choose_holdout_generators(
            _holdout_counts({"alpha": 100, "beta": 99, "gamma": 99})
        )


def test_choose_holdouts_allows_an_explicit_lower_fixture_threshold() -> None:
    selection = choose_holdout_generators(
        _holdout_counts({"alpha": 3, "beta": 2, "gamma": 1}),
        minimum_family_rows=2,
    )

    assert selection == HoldoutSelection(
        generator_validation_families=frozenset({"beta"}),
        generator_test_families=frozenset({"alpha"}),
    )


@pytest.mark.parametrize(
    ("validation", "test", "message"),
    [
        (frozenset(), frozenset({"alpha"}), "both be nonempty"),
        (frozenset({"alpha"}), frozenset({"alpha"}), "disjoint"),
        (frozenset({"missing"}), frozenset({"alpha"}), "not present"),
        (frozenset({"gamma"}), frozenset({"alpha"}), "at least 100"),
    ],
)
def test_manual_holdout_policy_is_preflight_validated(
    validation: frozenset[str], test: frozenset[str], message: str
) -> None:
    frame = _holdout_counts(
        {"alpha": 100, "beta": 100, "gamma": 99}, real_groups=20
    )
    policy = SplitPolicy(
        seed=17,
        validation_fraction=0.15,
        test_fraction=0.15,
        generator_validation_families=validation,
        generator_test_families=test,
    )

    with pytest.raises(DataIntegrityError, match=message):
        assign_grouped_splits(frame, policy)


def test_manual_holdout_policy_rejects_blank_family_identifiers() -> None:
    policy = SplitPolicy(
        seed=17,
        validation_fraction=0.15,
        test_fraction=0.15,
        generator_validation_families=frozenset({" "}),
        generator_test_families=frozenset({"alpha"}),
    )

    with pytest.raises(DataIntegrityError, match="nonempty.*family"):
        assign_grouped_splits(_assignable_frame(), policy)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"seed": True}, "seed"),
        ({"validation_fraction": "0.15"}, "validation_fraction"),
        ({"test_fraction": 0.0}, "test_fraction"),
        (
            {"validation_fraction": 0.60, "test_fraction": 0.40},
            "validation_fraction plus test_fraction",
        ),
        ({"generator_evaluation_real_fraction": 0.0}, "generator_evaluation_real_fraction"),
        ({"max_phash_distance": 65}, "max_phash_distance"),
        ({"minimum_holdout_family_rows": 0}, "minimum_holdout_family_rows"),
    ],
)
def test_split_policy_rejects_invalid_values_with_typed_preflight_errors(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(DataIntegrityError, match=message):
        assign_grouped_splits(
            _assignable_frame(),
            replace(_automatic_policy(), **changes),
        )


def test_split_policy_models_a_lower_threshold_for_tiny_fixtures() -> None:
    frame = _holdout_counts(
        {"alpha": 20, "beta": 19, "gamma": 18},
        real_groups=30,
    )
    policy = SplitPolicy(
        seed=17,
        validation_fraction=0.15,
        test_fraction=0.15,
        generator_validation_families=frozenset(),
        generator_test_families=frozenset(),
        generator_evaluation_real_fraction=0.10,
        minimum_holdout_family_rows=3,
    )

    split = assign_grouped_splits(frame, policy)

    assert set(split.loc[split.generator_family == "alpha", "split"]) == {
        "generator_test"
    }
    assert set(split.loc[split.generator_family == "beta", "split"]) == {
        "generator_validation"
    }


def test_duplicate_union_is_transitive_across_source_checksum_and_phash_chains() -> None:
    rows = [
        _row("a", source_group="source-chain", phash="0000000000000000"),
        _row("b", source_group="source-chain", phash="0000000000000003"),
        _row("c", checksum=_checksum("b"), phash="000000000000000f"),
        _row("d", phash="00000000000000ff"),
    ]
    frame = pd.DataFrame(rows)

    grouped = build_split_groups(frame, max_phash_distance=4)

    assert grouped["split_group_id"].nunique() == 1
    expected = "split-" + hashlib.sha256(b"a").hexdigest()[:16]
    assert grouped["split_group_id"].unique().tolist() == [expected]


def test_near_duplicate_phash_chain_collapses_transitively() -> None:
    frame = pd.DataFrame(
        [
            _row("a", phash="0000000000000000"),
            _row("b", phash="0000000000000003"),
            _row("c", phash="000000000000000f"),
        ]
    )

    grouped = build_split_groups(frame, max_phash_distance=2)

    assert (int("0", 16) ^ int("f", 16)).bit_count() > 2
    assert grouped["split_group_id"].nunique() == 1


@pytest.mark.parametrize("radius", [0, 1, 2, 4])
def test_bk_tree_radius_results_match_a_small_brute_force_oracle(radius: int) -> None:
    phashes = [
        "0000000000000000",
        "0000000000000001",
        "0000000000000003",
        "000000000000000f",
        "00000000000000ff",
        "8000000000000000",
        "c000000000000000",
        "ffffffffffffffff",
    ]
    frame = pd.DataFrame(
        [_row(f"sample:{index}", phash=value) for index, value in enumerate(phashes)]
    )

    grouped = build_split_groups(frame, max_phash_distance=radius)
    group_ids = grouped["split_group_id"].tolist()
    actual = {
        (first, second)
        for first in range(len(group_ids))
        for second in range(first + 1, len(group_ids))
        if group_ids[first] == group_ids[second]
    }

    assert actual == _brute_phash_connections(phashes, radius)


def test_hex_hash_case_is_normalized_for_duplicate_identity() -> None:
    checksum = _checksum("shared")
    frame = pd.DataFrame(
        [
            _row("a", checksum=checksum, phash="0000000000000000"),
            _row("b", checksum=checksum.upper(), phash="ffffffffffffffff"),
        ]
    )

    grouped = build_split_groups(frame, max_phash_distance=0)

    assert grouped["split_group_id"].nunique() == 1


def test_source_group_identity_remains_case_sensitive() -> None:
    frame = pd.DataFrame(
        [
            _row("a", source_group="Camera", phash="0000000000000000"),
            _row("b", source_group="camera", phash="ffffffffffffffff"),
        ]
    )

    grouped = build_split_groups(frame, max_phash_distance=0)

    assert grouped["split_group_id"].nunique() == 2


def test_split_group_ids_are_stable_under_row_reordering() -> None:
    frame = pd.DataFrame(
        [
            _row("z", checksum=_checksum("shared")),
            _row("a", checksum=_checksum("shared")),
            _row("m"),
        ]
    )

    forward = build_split_groups(frame, max_phash_distance=4).set_index("sample_id")
    reverse = build_split_groups(
        frame.iloc[::-1].reset_index(drop=True), max_phash_distance=4
    ).set_index("sample_id")

    assert forward["split_group_id"].to_dict() == reverse["split_group_id"].to_dict()


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("sample_id", None),
        ("source_group_id", ""),
        ("content_checksum", "not-sha256"),
        ("perceptual_hash", "1234"),
    ],
)
def test_build_split_groups_rejects_invalid_required_values_with_typed_error(
    column: str, bad_value: object
) -> None:
    frame = pd.DataFrame([_row("sample")])
    frame.loc[0, column] = bad_value

    with pytest.raises(DataIntegrityError, match=column):
        build_split_groups(frame, max_phash_distance=4)


def test_assert_no_leakage_allows_duplicates_inside_one_split() -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "train"],
            "content_checksum": ["same", "same"],
            "source_group_id": ["source", "source"],
            "split_group_id": ["group", "group"],
        }
    )

    assert_no_leakage(frame)


@pytest.mark.parametrize(
    ("column", "bad_value"),
    [
        ("content_checksum", None),
        ("content_checksum", "   "),
        ("source_group_id", pd.NA),
        ("source_group_id", ""),
        ("split_group_id", float("nan")),
        ("split_group_id", " "),
    ],
)
def test_assert_no_leakage_rejects_null_or_blank_identities(
    column: str, bad_value: object
) -> None:
    frame = pd.DataFrame(
        {
            "split": ["train"],
            "content_checksum": ["checksum"],
            "source_group_id": ["source"],
            "split_group_id": ["group"],
        }
    )
    frame.at[0, column] = bad_value

    with pytest.raises(DataIntegrityError, match=column):
        assert_no_leakage(frame)


@pytest.mark.parametrize(
    "column", ["content_checksum", "source_group_id", "split_group_id"]
)
def test_assert_no_leakage_rejects_identifier_crossing_splits(column: str) -> None:
    frame = pd.DataFrame(
        {
            "split": ["train", "test"],
            "content_checksum": ["checksum-a", "checksum-b"],
            "source_group_id": ["source-a", "source-b"],
            "split_group_id": ["group-a", "group-b"],
        }
    )
    frame.loc[:, column] = ["shared", "shared"]

    with pytest.raises(LeakageError, match=column):
        assert_no_leakage(frame)


def test_assert_no_leakage_normalizes_sha256_hex_case() -> None:
    checksum = _checksum("shared")
    frame = pd.DataFrame(
        {
            "split": ["train", "test"],
            "content_checksum": [checksum.lower(), checksum.upper()],
            "source_group_id": ["source-a", "source-b"],
            "split_group_id": ["group-a", "group-b"],
        }
    )

    with pytest.raises(LeakageError, match="content_checksum"):
        assert_no_leakage(frame)


def test_assign_grouped_splits_creates_exactly_five_leakage_safe_partitions() -> None:
    frame = _assignable_frame()
    # This non-holdout fake is content-identical to a holdout fake and must follow it.
    frame.loc[frame.sample_id == "gamma:000", "content_checksum"] = frame.loc[
        frame.sample_id == "alpha:000", "content_checksum"
    ].iloc[0]

    split = assign_grouped_splits(frame, _automatic_policy())

    assert set(split["split"]) == {
        "train",
        "validation",
        "test",
        "generator_validation",
        "generator_test",
    }
    assert set(split.loc[split.generator_family == "alpha", "split"]) == {
        "generator_test"
    }
    assert set(split.loc[split.generator_family == "beta", "split"]) == {
        "generator_validation"
    }
    assert split.loc[split.sample_id == "gamma:000", "split"].item() == "generator_test"
    assert all(set(partition.label) == {0, 1} for _, partition in split.groupby("split"))
    assert split.groupby("source_group_id")["split"].nunique().max() == 1
    assert split.groupby("content_checksum")["split"].nunique().max() == 1
    assert split.groupby("split_group_id")["split"].nunique().max() == 1
    generator_validation_real_groups = set(
        split.loc[
            (split.label == 0) & (split.split == "generator_validation"),
            "split_group_id",
        ]
    )
    generator_test_real_groups = set(
        split.loc[
            (split.label == 0) & (split.split == "generator_test"),
            "split_group_id",
        ]
    )
    assert generator_validation_real_groups
    assert generator_test_real_groups
    assert generator_validation_real_groups.isdisjoint(generator_test_real_groups)


def test_paired_holdout_groups_supply_their_own_real_controls() -> None:
    rows: list[dict[str, object]] = []
    for family in ("alpha", "beta", "gamma"):
        for index in range(30):
            group = f"pair:{family}:{index}"
            rows.append(_row(f"real:{family}:{index}", source_group=group))
            rows.append(
                _row(
                    f"fake:{family}:{index}",
                    label=1,
                    family=family,
                    source_group=group,
                )
            )
    policy = replace(_automatic_policy(), minimum_holdout_family_rows=20)

    split = assign_grouped_splits(pd.DataFrame(rows), policy)

    assert set(split["split"]) == set(PARTITIONS)
    assert all(set(partition.label) == {0, 1} for _, partition in split.groupby("split"))
    assert split.groupby("source_group_id")["split"].nunique().max() == 1


def test_assign_grouped_splits_rejects_a_component_linking_both_holdouts() -> None:
    frame = _assignable_frame()
    shared = frame.loc[frame.sample_id == "alpha:000", "content_checksum"].iloc[0]
    frame.loc[frame.sample_id == "beta:000", "content_checksum"] = shared

    with pytest.raises(LeakageError, match="links generator validation and generator test"):
        assign_grouped_splits(frame, _automatic_policy())


def test_assign_grouped_splits_is_deterministic_under_row_reordering() -> None:
    frame = _assignable_frame()
    original = frame.copy(deep=True)
    policy = _automatic_policy(seed=29)

    forward = assign_grouped_splits(frame, policy).set_index("sample_id")["split"]
    reverse = assign_grouped_splits(
        frame.sample(frac=1.0, random_state=3).reset_index(drop=True), policy
    ).set_index("sample_id")["split"]

    assert forward.to_dict() == reverse.to_dict()
    pd.testing.assert_frame_equal(frame, original)


def test_holdout_family_completeness_applies_only_to_fake_rows() -> None:
    frame = _assignable_frame()
    frame.loc[frame.label == 0, "generator_family"] = "alpha"

    split = assign_grouped_splits(frame, _automatic_policy())

    assert set(split.loc[split.label == 0, "split"]) == set(PARTITIONS)
    assert set(
        split.loc[
            (split.label == 1) & (split.generator_family == "alpha"), "split"
        ]
    ) == {"generator_test"}


def test_assign_grouped_splits_rejects_too_few_real_groups_before_sampling() -> None:
    frame = _holdout_counts(
        {"alpha": 100, "beta": 100, "gamma": 100}, real_groups=1
    )

    with pytest.raises(DataIntegrityError, match="real groups.*found 1"):
        assign_grouped_splits(frame, _automatic_policy())


def test_assign_grouped_splits_reports_class_and_group_counts_when_partitioning_is_impossible() -> None:
    frame = _assignable_frame()
    # Collapse every remaining non-holdout fake into one source group. It cannot supply
    # fake examples to train, validation, and test simultaneously.
    frame.loc[frame.generator_family == "gamma", "source_group_id"] = "gamma:one-group"

    with pytest.raises(DataIntegrityError, match=r"class counts=.*group counts="):
        assign_grouped_splits(frame, _automatic_policy())


def test_split_writer_round_trips_deterministic_parquet_and_metadata(tmp_path: Path) -> None:
    source_manifest = tmp_path / "source.parquet"
    source_manifest.write_bytes(b"canonical source manifest bytes")
    output = tmp_path / "assigned.parquet"
    frame = _assignable_frame()
    policy = _automatic_policy()

    result = write_split_manifest(frame, output, policy, source_manifest)
    first_parquet = output.read_bytes()
    first_metadata = result.metadata_path.read_bytes()
    readback = pd.read_parquet(output)
    metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))

    assert split_metadata_path(output) == output.with_suffix(".json")
    assert set(readback["split"]) == {
        "train",
        "validation",
        "test",
        "generator_validation",
        "generator_test",
    }
    assert metadata["policy"]["seed"] == 17
    assert metadata["policy"]["minimum_holdout_family_rows"] == 100
    assert metadata["chosen_holdouts"] == {
        "generator_test_families": ["alpha"],
        "generator_validation_families": ["beta"],
    }
    assert metadata["row_counts"] == readback["split"].value_counts().sort_index().to_dict()
    assert metadata["group_counts"] == (
        readback.groupby("split")["split_group_id"].nunique().sort_index().to_dict()
    )
    assert metadata["source_manifest_sha256"] == hashlib.sha256(
        source_manifest.read_bytes()
    ).hexdigest()
    assert metadata["split_sha256"] == result.split_sha256
    assert result.split_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert len(result.split_sha256) == 64
    assert metadata["source_manifest_sha256"] != metadata["split_sha256"]

    reordered = frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    second = write_split_manifest(reordered, output, replace(policy), source_manifest)

    assert output.read_bytes() == first_parquet
    assert second.metadata_path.read_bytes() == first_metadata
    assert second.split_sha256 == result.split_sha256


def test_split_writer_rejects_a_non_parquet_destination(tmp_path: Path) -> None:
    source_manifest = tmp_path / "source.parquet"
    source_manifest.write_bytes(b"source")
    output = tmp_path / "assigned.txt"

    with pytest.raises(DataIntegrityError, match="Parquet"):
        write_split_manifest(
            _assignable_frame(),
            output,
            _automatic_policy(),
            source_manifest,
        )

    assert not output.exists()


def test_split_writer_preserves_existing_outputs_when_serialization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest = tmp_path / "source.parquet"
    source_manifest.write_bytes(b"source")
    output = tmp_path / "assigned.parquet"
    metadata_path = split_metadata_path(output)
    output.write_bytes(b"old parquet")
    metadata_path.write_bytes(b"old metadata")

    def fail_write(self: pd.DataFrame, *args: object, **kwargs: object) -> None:
        raise OSError("disk failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)

    with pytest.raises(OSError, match="disk failure"):
        write_split_manifest(_assignable_frame(), output, _automatic_policy(), source_manifest)

    assert output.read_bytes() == b"old parquet"
    assert metadata_path.read_bytes() == b"old metadata"
    assert not list(tmp_path.glob(".*.tmp"))


def test_split_writer_restores_both_outputs_when_metadata_backup_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_manifest = tmp_path / "source.parquet"
    source_manifest.write_bytes(b"source")
    output = tmp_path / "assigned.parquet"
    metadata_path = split_metadata_path(output)
    output.write_bytes(b"old parquet")
    metadata_path.write_bytes(b"old metadata")
    original_replace = Path.replace
    injected = False

    def fail_metadata_backup(self: Path, target: str | Path) -> Path:
        nonlocal injected
        destination = Path(target)
        if self == metadata_path and destination.name.endswith(".backup") and not injected:
            injected = True
            raise OSError("metadata backup failure")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", fail_metadata_backup)

    with pytest.raises(OSError, match="metadata backup failure"):
        write_split_manifest(
            _assignable_frame(),
            output,
            _automatic_policy(),
            source_manifest,
        )

    assert output.read_bytes() == b"old parquet"
    assert metadata_path.read_bytes() == b"old metadata"
    assert not list(tmp_path.glob(".*.tmp"))
    assert not list(tmp_path.glob(".*.backup"))
