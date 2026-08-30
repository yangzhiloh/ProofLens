from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from prooflens.data.hashing import sha256_file
from prooflens.errors import DataIntegrityError, LeakageError

PARTITIONS = (
    "train",
    "validation",
    "test",
    "generator_validation",
    "generator_test",
)
_GROUP_COLUMNS = (
    "sample_id",
    "source_group_id",
    "content_checksum",
    "perceptual_hash",
)
_LEAKAGE_COLUMNS = ("content_checksum", "source_group_id", "split_group_id")
_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
_PHASH_PATTERN = re.compile(r"^[0-9a-fA-F]{16}$")
_PRODUCTION_MINIMUM_HOLDOUT_ROWS = 100


@dataclass(frozen=True)
class HoldoutSelection:
    generator_validation_families: frozenset[str]
    generator_test_families: frozenset[str]


@dataclass(frozen=True)
class SplitPolicy:
    seed: int
    validation_fraction: float
    test_fraction: float
    generator_validation_families: frozenset[str]
    generator_test_families: frozenset[str]
    generator_evaluation_real_fraction: float = 0.10
    max_phash_distance: int = 4
    minimum_holdout_family_rows: int = _PRODUCTION_MINIMUM_HOLDOUT_ROWS


@dataclass(frozen=True)
class SplitWriteResult:
    output_path: Path
    metadata_path: Path
    split_sha256: str


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != item:
            parent = self.parent[item]
            self.parent[item] = root
            item = parent
        return root

    def union(self, first: int, second: int) -> None:
        left = self.find(first)
        right = self.find(second)
        if left == right:
            return
        if self.rank[left] < self.rank[right]:
            left, right = right, left
        self.parent[right] = left
        if self.rank[left] == self.rank[right]:
            self.rank[left] += 1


@dataclass
class _BKNode:
    value: int
    row_indices: list[int]
    children: dict[int, _BKNode]


class _BKTree:
    """Exact metric index for radius queries over 64-bit Hamming distance."""

    def __init__(self) -> None:
        self.root: _BKNode | None = None

    def add(self, value: int, row_index: int) -> None:
        if self.root is None:
            self.root = _BKNode(value=value, row_indices=[row_index], children={})
            return
        node = self.root
        while True:
            distance = _hamming_distance(value, node.value)
            if distance == 0:
                node.row_indices.append(row_index)
                return
            child = node.children.get(distance)
            if child is None:
                node.children[distance] = _BKNode(
                    value=value, row_indices=[row_index], children={}
                )
                return
            node = child

    def neighbors(self, value: int, radius: int) -> tuple[int, ...]:
        if self.root is None:
            return ()
        matches: list[int] = []
        pending = [self.root]
        while pending:
            node = pending.pop()
            distance = _hamming_distance(value, node.value)
            if distance <= radius:
                matches.extend(node.row_indices)
            minimum = max(0, distance - radius)
            maximum = distance + radius
            pending.extend(
                child
                for edge, child in node.children.items()
                if minimum <= edge <= maximum
            )
        return tuple(matches)


def _hamming_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def choose_holdout_generators(
    frame: pd.DataFrame,
    *,
    minimum_family_rows: int = _PRODUCTION_MINIMUM_HOLDOUT_ROWS,
) -> HoldoutSelection:
    """Choose the two largest eligible fake families with a lexical tie-break."""
    _validate_positive_integer(
        minimum_family_rows,
        "minimum_family_rows",
    )
    counts = _fake_family_counts(frame)
    if len(counts) < 3:
        raise DataIntegrityError(
            f"generator holdout selection requires at least three fake families; found {len(counts)}"
        )
    eligible = [
        (family, count) for family, count in counts.items() if count >= minimum_family_rows
    ]
    if len(eligible) < 2:
        rendered = ", ".join(f"{family}={count}" for family, count in counts.items())
        raise DataIntegrityError(
            "generator holdout selection requires at least two families with at least "
            f"{minimum_family_rows} rows; found {rendered}"
        )
    ranked = sorted(eligible, key=lambda item: (-item[1], item[0]))
    generator_test = ranked[0][0]
    generator_validation = ranked[1][0]
    return HoldoutSelection(
        generator_validation_families=frozenset({generator_validation}),
        generator_test_families=frozenset({generator_test}),
    )


def build_split_groups(frame: pd.DataFrame, max_phash_distance: int) -> pd.DataFrame:
    """Union source, exact-byte, and pHash-neighbor relationships transitively."""
    if (
        not isinstance(max_phash_distance, int)
        or isinstance(max_phash_distance, bool)
        or not 0 <= max_phash_distance <= 64
    ):
        raise DataIntegrityError("max_phash_distance must be an integer from 0 to 64")
    _require_columns(frame, _GROUP_COLUMNS, "duplicate grouping")
    result = frame.copy(deep=True)
    if result.empty:
        result["split_group_id"] = pd.Series(dtype="string")
        return result
    sample_ids = _validated_strings(result, "sample_id", unique=True)
    source_groups = _validated_strings(result, "source_group_id")
    checksums = _validated_hashes(result, "content_checksum", _SHA256_PATTERN, "SHA-256")
    phashes = _validated_hashes(result, "perceptual_hash", _PHASH_PATTERN, "64-bit pHash")

    union = _UnionFind(len(result))
    _union_equal_values(source_groups, union)
    _union_equal_values(checksums, union)

    tree = _BKTree()
    indexed_hashes = sorted(
        ((int(value, 16), sample_ids[index], index) for index, value in enumerate(phashes)),
        key=lambda item: (item[0], item[1]),
    )
    for value, _, row_index in indexed_hashes:
        for neighbor in tree.neighbors(value, max_phash_distance):
            union.union(row_index, neighbor)
        tree.add(value, row_index)

    components: dict[int, list[int]] = {}
    for index in range(len(result)):
        components.setdefault(union.find(index), []).append(index)
    group_id_by_row: dict[int, str] = {}
    for row_indices in components.values():
        minimum_sample_id = min(sample_ids[index] for index in row_indices)
        group_id = "split-" + hashlib.sha256(minimum_sample_id.encode("utf-8")).hexdigest()[:16]
        group_id_by_row.update({index: group_id for index in row_indices})
    result["split_group_id"] = [group_id_by_row[index] for index in range(len(result))]
    return result


def assert_no_leakage(frame: pd.DataFrame) -> None:
    """Reject invalid identities and identifiers observed in more than one split."""
    _require_columns(frame, ("split", *_LEAKAGE_COLUMNS), "leakage validation")
    splits = _validated_strings(frame, "split")
    for column in _LEAKAGE_COLUMNS:
        identities = _validated_strings(frame, column)
        if column == "content_checksum":
            identities = [identity.casefold() for identity in identities]
        observed = pd.DataFrame({column: identities, "split": splits})
        counts = observed.groupby(column, sort=True)["split"].nunique()
        leaking = counts[counts > 1]
        if not leaking.empty:
            examples = ", ".join(str(value) for value in leaking.index[:3])
            raise LeakageError(
                f"{column} appears in multiple splits; leaking identifiers: {examples}"
            )


def assign_grouped_splits(frame: pd.DataFrame, policy: SplitPolicy) -> pd.DataFrame:
    """Create deterministic train, validation, test, and generator holdouts."""
    _validate_policy(policy)
    _require_columns(frame, ("label", "generator_family"), "split assignment")
    labels = _validated_labels(frame)
    _validated_strings(frame, "generator_family")
    if set(labels) != {0, 1}:
        raise DataIntegrityError(
            f"split assignment requires both labels; class counts={_class_counts(frame)}"
        )
    selection = _resolve_holdouts(frame, policy)
    result = build_split_groups(frame, policy.max_phash_distance)
    result["split"] = "unassigned"

    validation_groups = set(
        result.loc[
            (result["label"] == 1)
            & result["generator_family"].isin(selection.generator_validation_families),
            "split_group_id",
        ]
    )
    test_groups = set(
        result.loc[
            (result["label"] == 1)
            & result["generator_family"].isin(selection.generator_test_families),
            "split_group_id",
        ]
    )
    overlap = validation_groups & test_groups
    if overlap:
        raise LeakageError(
            "a connected duplicate or source group links generator validation and generator test"
        )
    result.loc[result["split_group_id"].isin(validation_groups), "split"] = (
        "generator_validation"
    )
    result.loc[result["split_group_id"].isin(test_groups), "split"] = "generator_test"

    holdouts_missing_real = [
        split
        for split in ("generator_validation", "generator_test")
        if not (result.loc[result["split"] == split, "label"] == 0).any()
    ]
    real_only_groups = sorted(
        group_id
        for group_id, group in result[result["split"] == "unassigned"].groupby(
            "split_group_id", sort=True
        )
        if set(group["label"].astype(int)) == {0}
    )
    real_count = (
        max(
            1,
            int(
                np.floor(
                    len(real_only_groups)
                    * policy.generator_evaluation_real_fraction
                    + 0.5
                )
            ),
        )
        if holdouts_missing_real
        else 0
    )
    required_real_groups = len(holdouts_missing_real) * real_count
    if len(real_only_groups) < required_real_groups:
        raise DataIntegrityError(
            "generator evaluation requires disjoint real groups for both holdouts; "
            f"needed {required_real_groups}, found {len(real_only_groups)} real groups"
        )
    rng = np.random.default_rng(policy.seed)
    chosen_real = np.asarray(real_only_groups, dtype=object)[rng.permutation(len(real_only_groups))]
    for index, split in enumerate(holdouts_missing_real):
        start = index * real_count
        selected_real = set(chosen_real[start : start + real_count])
        result.loc[result["split_group_id"].isin(selected_real), "split"] = split

    remaining = result[result["split"] == "unassigned"]
    assignments = _three_way_group_assignment(remaining, policy)
    result.loc[result["split_group_id"].isin(assignments["train"]), "split"] = "train"
    result.loc[result["split_group_id"].isin(assignments["validation"]), "split"] = (
        "validation"
    )
    result.loc[result["split_group_id"].isin(assignments["test"]), "split"] = "test"

    _assert_complete_holdout_families(result, selection)
    _assert_partition_labels(result)
    assert_no_leakage(result)
    return result


def split_metadata_path(output_path: Path) -> Path:
    return Path(output_path).with_suffix(".json")


def write_split_manifest(
    frame: pd.DataFrame,
    output_path: Path,
    policy: SplitPolicy,
    source_manifest_path: Path,
) -> SplitWriteResult:
    """Assign and atomically persist canonical Parquet plus adjacent JSON metadata."""
    output_path = Path(output_path)
    source_manifest_path = Path(source_manifest_path)
    if output_path.suffix.casefold() != ".parquet":
        raise DataIntegrityError("split destination must be a Parquet file with a .parquet suffix")
    if not source_manifest_path.is_file():
        raise DataIntegrityError(f"source manifest is not a file: {source_manifest_path}")
    assigned = assign_grouped_splits(frame, policy)
    canonical = assigned.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    selection = _resolve_holdouts(canonical, policy)
    source_manifest_sha256 = sha256_file(source_manifest_path)
    metadata_path = split_metadata_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    parquet_temporary = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    metadata_temporary = metadata_path.with_name(f".{metadata_path.name}.{uuid4().hex}.tmp")
    try:
        canonical.to_parquet(parquet_temporary, index=False)
        split_sha256 = sha256_file(parquet_temporary)
        metadata = {
            "policy": _policy_payload(policy),
            "chosen_holdouts": _selection_payload(selection),
            "row_counts": {
                str(key): int(value)
                for key, value in canonical["split"].value_counts().sort_index().items()
            },
            "group_counts": {
                str(key): int(value)
                for key, value in canonical.groupby("split", sort=True)["split_group_id"]
                .nunique()
                .items()
            },
            "source_manifest_sha256": source_manifest_sha256,
            "split_sha256": split_sha256,
        }
        metadata_text = json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n"
        metadata_temporary.write_text(metadata_text, encoding="utf-8")
        _replace_pair_atomically(
            parquet_temporary,
            output_path,
            metadata_temporary,
            metadata_path,
        )
    finally:
        for temporary in (parquet_temporary, metadata_temporary):
            if temporary.exists():
                temporary.unlink()
    return SplitWriteResult(output_path, metadata_path, split_sha256)


def _resolve_holdouts(frame: pd.DataFrame, policy: SplitPolicy) -> HoldoutSelection:
    validation = frozenset(policy.generator_validation_families)
    test = frozenset(policy.generator_test_families)
    if not validation and not test:
        return choose_holdout_generators(
            frame,
            minimum_family_rows=policy.minimum_holdout_family_rows,
        )
    if not validation or not test:
        raise DataIntegrityError(
            "manual generator validation and test families must both be nonempty"
        )
    overlap = validation & test
    if overlap:
        raise DataIntegrityError(
            f"manual generator validation and test families must be disjoint: {sorted(overlap)}"
        )
    counts = _fake_family_counts(frame)
    if len(counts) < 3:
        raise DataIntegrityError(
            f"generator holdout policy requires at least three fake families; found {len(counts)}"
        )
    missing = (validation | test) - set(counts)
    if missing:
        raise DataIntegrityError(
            f"manual generator holdout families are not present in fake rows: {sorted(missing)}"
        )
    too_small = {
        family: counts[family]
        for family in sorted(validation | test)
        if counts[family] < policy.minimum_holdout_family_rows
    }
    if too_small:
        rendered = ", ".join(f"{family}={count}" for family, count in too_small.items())
        raise DataIntegrityError(
            "manual generator holdout families require at least "
            f"{policy.minimum_holdout_family_rows} rows; found {rendered}"
        )
    return HoldoutSelection(validation, test)


def _three_way_group_assignment(
    frame: pd.DataFrame, policy: SplitPolicy
) -> dict[str, set[str]]:
    groups = sorted(frame["split_group_id"].unique())
    if len(groups) < 3:
        raise _partition_error(frame, "at least three remaining groups are required")
    test_count = max(1, int(np.floor(len(groups) * policy.test_fraction + 0.5)))
    validation_count = max(
        1, int(np.floor(len(groups) * policy.validation_fraction + 0.5))
    )
    if test_count + validation_count >= len(groups):
        raise _partition_error(frame, "split fractions leave no training group")
    values = np.asarray(groups, dtype=object)
    for offset in range(100):
        permutation = np.random.default_rng(policy.seed + offset).permutation(len(groups))
        shuffled = values[permutation]
        test = set(shuffled[:test_count])
        validation = set(shuffled[test_count : test_count + validation_count])
        train = set(shuffled[test_count + validation_count :])
        assignments = {"train": train, "validation": validation, "test": test}
        if all(_groups_have_both_labels(frame, selected) for selected in assignments.values()):
            return assignments
    raise _partition_error(frame, "100 deterministic seeds could not preserve both labels")


def _groups_have_both_labels(frame: pd.DataFrame, selected: set[str]) -> bool:
    return set(frame.loc[frame["split_group_id"].isin(selected), "label"].astype(int)) == {0, 1}


def _partition_error(frame: pd.DataFrame, reason: str) -> DataIntegrityError:
    group_counts = (
        frame.groupby("label", sort=True)["split_group_id"].nunique().astype(int).to_dict()
    )
    return DataIntegrityError(
        f"cannot group-split remaining rows: {reason}; "
        f"class counts={_class_counts(frame)}, group counts={group_counts}"
    )


def _assert_complete_holdout_families(
    frame: pd.DataFrame, selection: HoldoutSelection
) -> None:
    expected = (
        (selection.generator_validation_families, "generator_validation"),
        (selection.generator_test_families, "generator_test"),
    )
    for families, split in expected:
        observed = set(
            frame.loc[
                (frame["label"] == 1) & frame["generator_family"].isin(families),
                "split",
            ]
        )
        if observed != {split}:
            raise LeakageError(
                f"complete fake holdout families {sorted(families)} must occur only in {split}"
            )


def _assert_partition_labels(frame: pd.DataFrame) -> None:
    observed = set(frame["split"])
    if observed != set(PARTITIONS):
        raise DataIntegrityError(
            f"split assignment must create exactly five partitions; found {sorted(observed)}"
        )
    invalid = {
        split: _class_counts(group)
        for split, group in frame.groupby("split", sort=True)
        if set(group["label"].astype(int)) != {0, 1}
    }
    if invalid:
        raise DataIntegrityError(
            f"every partition must contain both labels; diagnostic class counts={invalid}"
        )


def _validate_policy(policy: SplitPolicy) -> None:
    if not isinstance(policy, SplitPolicy):
        raise DataIntegrityError("split policy must be a SplitPolicy")
    if (
        not isinstance(policy.seed, Integral)
        or isinstance(policy.seed, bool)
        or int(policy.seed) < 0
    ):
        raise DataIntegrityError("seed must be a nonnegative integer")
    for name, value in (
        ("validation_fraction", policy.validation_fraction),
        ("test_fraction", policy.test_fraction),
    ):
        if (
            not isinstance(value, Real)
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            or not 0 < float(value) < 1
        ):
            raise DataIntegrityError(f"{name} must be between 0 and 1")
    if float(policy.validation_fraction) + float(policy.test_fraction) >= 1:
        raise DataIntegrityError("validation_fraction plus test_fraction must be less than 1")
    real_fraction = policy.generator_evaluation_real_fraction
    if (
        not isinstance(real_fraction, Real)
        or isinstance(real_fraction, bool)
        or not math.isfinite(float(real_fraction))
        or not 0 < float(real_fraction) <= 0.5
    ):
        raise DataIntegrityError(
            "generator_evaluation_real_fraction must be greater than 0 and at most 0.5"
        )
    if (
        not isinstance(policy.max_phash_distance, Integral)
        or isinstance(policy.max_phash_distance, bool)
        or not 0 <= int(policy.max_phash_distance) <= 64
    ):
        raise DataIntegrityError("max_phash_distance must be an integer from 0 to 64")
    _validate_positive_integer(
        policy.minimum_holdout_family_rows,
        "minimum_holdout_family_rows",
    )
    _validated_policy_families(
        policy.generator_validation_families,
        "generator_validation_families",
    )
    _validated_policy_families(
        policy.generator_test_families,
        "generator_test_families",
    )


def _validate_positive_integer(value: object, field: str) -> None:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 1:
        raise DataIntegrityError(f"{field} must be a positive integer")


def _validated_policy_families(value: object, field: str) -> frozenset[str]:
    if not isinstance(value, frozenset):
        raise DataIntegrityError(f"{field} must be a frozenset of nonempty family names")
    if any(not isinstance(family, str) or not family.strip() for family in value):
        raise DataIntegrityError(f"{field} must contain only nonempty family names")
    return value


def _fake_family_counts(frame: pd.DataFrame) -> dict[str, int]:
    _require_columns(frame, ("label", "generator_family"), "generator holdout selection")
    labels = _validated_labels(frame)
    families = _validated_strings(frame, "generator_family")
    fake = pd.DataFrame({"label": labels, "family": families})
    counts = fake.loc[fake["label"] == 1, "family"].value_counts()
    return {
        str(family): int(count)
        for family, count in sorted(counts.items(), key=lambda item: item[0])
    }


def _validated_labels(frame: pd.DataFrame) -> list[int]:
    labels: list[int] = []
    for index, value in frame["label"].items():
        if isinstance(value, bool) or pd.isna(value):
            raise DataIntegrityError(f"label at row {index} must be binary 0 or 1")
        try:
            label = int(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise DataIntegrityError(f"label at row {index} must be binary 0 or 1") from error
        if label not in (0, 1) or float(value) != label:
            raise DataIntegrityError(f"label at row {index} must be binary 0 or 1")
        labels.append(label)
    return labels


def _validated_strings(
    frame: pd.DataFrame, column: str, *, unique: bool = False
) -> list[str]:
    values: list[str] = []
    for index, value in frame[column].items():
        if not isinstance(value, str) or not value.strip():
            raise DataIntegrityError(f"{column} at row {index} must be a nonempty string")
        values.append(value)
    if unique and len(values) != len(set(values)):
        duplicates = sorted(value for value, count in Counter(values).items() if count > 1)
        raise DataIntegrityError(f"{column} values must be unique; duplicates: {duplicates[:3]}")
    return values


def _validated_hashes(
    frame: pd.DataFrame, column: str, pattern: re.Pattern[str], description: str
) -> list[str]:
    values: list[str] = []
    for index, value in frame[column].items():
        if not isinstance(value, str) or pattern.fullmatch(value) is None:
            raise DataIntegrityError(
                f"{column} at row {index} must be a hexadecimal {description}"
            )
        values.append(value.lower())
    return values


def _union_equal_values(values: list[str], union: _UnionFind) -> None:
    first_index: dict[str, int] = {}
    for index, value in enumerate(values):
        previous = first_index.setdefault(value, index)
        union.union(previous, index)


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], operation: str) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise DataIntegrityError(f"{operation} requires a pandas DataFrame")
    missing = set(columns) - set(frame.columns)
    if missing:
        raise DataIntegrityError(f"{operation} is missing required fields: {sorted(missing)}")


def _class_counts(frame: pd.DataFrame) -> dict[int, int]:
    return {
        int(label): int(count)
        for label, count in frame["label"].value_counts().sort_index().items()
    }


def _policy_payload(policy: SplitPolicy) -> dict[str, Any]:
    return {
        "seed": int(policy.seed),
        "validation_fraction": float(policy.validation_fraction),
        "test_fraction": float(policy.test_fraction),
        "generator_validation_families": sorted(policy.generator_validation_families),
        "generator_test_families": sorted(policy.generator_test_families),
        "generator_evaluation_real_fraction": float(
            policy.generator_evaluation_real_fraction
        ),
        "max_phash_distance": int(policy.max_phash_distance),
        "minimum_holdout_family_rows": int(policy.minimum_holdout_family_rows),
    }


def _selection_payload(selection: HoldoutSelection) -> dict[str, list[str]]:
    return {
        "generator_validation_families": sorted(
            selection.generator_validation_families
        ),
        "generator_test_families": sorted(selection.generator_test_families),
    }


def _replace_pair_atomically(
    first_temporary: Path,
    first_destination: Path,
    second_temporary: Path,
    second_destination: Path,
) -> None:
    first_backup = first_destination.with_name(
        f".{first_destination.name}.{uuid4().hex}.backup"
    )
    second_backup = second_destination.with_name(
        f".{second_destination.name}.{uuid4().hex}.backup"
    )
    first_existed = first_destination.exists()
    second_existed = second_destination.exists()
    first_backed_up = False
    second_backed_up = False
    first_published = False
    second_published = False
    try:
        if first_existed:
            first_destination.replace(first_backup)
            first_backed_up = True
        if second_existed:
            second_destination.replace(second_backup)
            second_backed_up = True
        first_temporary.replace(first_destination)
        first_published = True
        second_temporary.replace(second_destination)
        second_published = True
    except Exception:
        if first_backed_up:
            first_backup.replace(first_destination)
        elif first_published and first_destination.exists():
            first_destination.unlink()
        if second_backed_up:
            second_backup.replace(second_destination)
        elif second_published and second_destination.exists():
            second_destination.unlink()
        raise
    for backup in (first_backup, second_backup):
        if backup.exists():
            backup.unlink()
