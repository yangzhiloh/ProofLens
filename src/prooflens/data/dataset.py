from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from numbers import Integral
from os import PathLike
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import Dataset

from prooflens.data.schema import MANIFEST_COLUMNS
from prooflens.errors import DataIntegrityError, ImageDecodeError

_ASSIGNED_SPLITS = frozenset(
    {"train", "validation", "test", "generator_validation", "generator_test"}
)
_REQUIRED_COLUMNS = (*MANIFEST_COLUMNS, "split_group_id")
_REQUIRED_TEXT = (
    "sample_id",
    "dataset_name",
    "generator_family",
    "source_group_id",
    "split_group_id",
)


@dataclass(frozen=True, slots=True)
class SourceItem:
    """One decoded source image and the manifest identity needed downstream."""

    image: Image.Image
    label: int
    sample_id: str
    dataset_name: str
    generator_family: str
    source_group_id: str
    split: str
    split_group_id: str


@dataclass(frozen=True, slots=True)
class _SourceRecord:
    path: Path
    label: int
    sample_id: str
    dataset_name: str
    generator_family: str
    source_group_id: str
    split: str
    split_group_id: str


class SourceImageDataset(Dataset[SourceItem]):
    """Decode an immutable positional snapshot of an assigned canonical manifest."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._records = _validated_records(frame)

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> SourceItem:
        record = self._records[index]
        image = _decode_rgb(record.path, record.sample_id)
        return SourceItem(
            image=image,
            label=record.label,
            sample_id=record.sample_id,
            dataset_name=record.dataset_name,
            generator_family=record.generator_family,
            source_group_id=record.source_group_id,
            split=record.split,
            split_group_id=record.split_group_id,
        )


def _validated_records(frame: pd.DataFrame) -> tuple[_SourceRecord, ...]:
    if not isinstance(frame, pd.DataFrame):
        raise DataIntegrityError("source dataset requires a pandas DataFrame")
    missing = set(_REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise DataIntegrityError(
            f"source dataset is missing required fields: {sorted(missing)}"
        )

    records: list[_SourceRecord] = []
    sample_ids: list[str] = []
    for position in range(len(frame)):
        row = frame.iloc[position]
        text = {
            field: _nonempty_text(row[field], field, position)
            for field in _REQUIRED_TEXT
        }
        label = _binary_label(row["label"], position)
        path = _local_path(row["path"], position)
        split = _nonempty_text(row["split"], "split", position)
        if split not in _ASSIGNED_SPLITS:
            raise DataIntegrityError(
                f"split at row position {position} must be an assigned split"
            )
        sample_ids.append(text["sample_id"])
        records.append(
            _SourceRecord(
                path=path,
                label=label,
                sample_id=text["sample_id"],
                dataset_name=text["dataset_name"],
                generator_family=text["generator_family"],
                source_group_id=text["source_group_id"],
                split=split,
                split_group_id=text["split_group_id"],
            )
        )
    duplicates = sorted(
        value for value, count in Counter(sample_ids).items() if count > 1
    )
    if duplicates:
        raise DataIntegrityError(
            f"source dataset sample_id values must be unique; duplicates: {duplicates[:3]}"
        )
    return tuple(records)


def _binary_label(value: object, position: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) not in (0, 1):
        raise DataIntegrityError(
            f"label at row position {position} must be binary 0 or 1"
        )
    return int(value)


def _nonempty_text(value: object, field: str, position: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataIntegrityError(
            f"{field} at row position {position} must be a nonempty string"
        )
    return value


def _local_path(value: object, position: int) -> Path:
    if isinstance(value, str):
        if not value.strip():
            raise DataIntegrityError(
                f"path at row position {position} must be a nonempty local path"
            )
    elif not isinstance(value, PathLike):
        raise DataIntegrityError(
            f"path at row position {position} must be a nonempty local path"
        )
    try:
        return Path(value)
    except (OSError, TypeError, ValueError) as error:
        raise DataIntegrityError(
            f"path at row position {position} must be a nonempty local path"
        ) from error


def _decode_rgb(path: Path, sample_id: str) -> Image.Image:
    try:
        with Image.open(path) as verifier:
            verifier.verify()
        with Image.open(path) as source:
            transposed = ImageOps.exif_transpose(source)
            transposed.load()
            rgb = transposed.convert("RGB")
            rgb.load()
            return rgb.copy()
    except (
        Image.DecompressionBombError,
        OSError,
        SyntaxError,
        TypeError,
        UnidentifiedImageError,
        ValueError,
    ) as error:
        raise ImageDecodeError(
            f"cannot decode sample {sample_id!r} at {path}: {error}"
        ) from error
