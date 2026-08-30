from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from uuid import uuid4

import pandas as pd
import yaml
from PIL import Image

from prooflens.data.adapters.sid_set import SidSetAdapter
from prooflens.data.licences import (
    SID_SET,
    wildfake_manual_acquisition_message,
)
from prooflens.data.manifest import build_manifest
from prooflens.data.schema import ManifestRecord
from prooflens.errors import DatasetAcquisitionError, DatasetPolicyError, UserInputError

SID_SET_PINNED_REVISION = "c1674903d858c78e04809c1c6f2703627ac1a621"
ACQUISITION_METADATA_NAME = "acquisition.json"


@dataclass(frozen=True)
class SidAcquisitionConfig:
    dataset_id: str = "saberzl/SID_Set"
    split: str = "validation"
    streaming: bool = True
    revision: str = SID_SET_PINNED_REVISION
    per_class: int = 10_000
    image_field: str = "image"
    id_field: str = "img_id"
    label_field: str = "label"
    images_directory: str = "images"
    manifest_name: str = "manifest.parquet"

    @classmethod
    def from_value(
        cls, value: SidAcquisitionConfig | Mapping[str, Any] | Path
    ) -> SidAcquisitionConfig:
        if isinstance(value, cls):
            result = value
        else:
            raw = (
                yaml.safe_load(value.read_text(encoding="utf-8"))
                if isinstance(value, Path)
                else dict(value)
            )
            if not isinstance(raw, Mapping):
                raise UserInputError("SID acquisition configuration must be a mapping")
            unknown = set(raw) - set(cls.__dataclass_fields__)
            if unknown:
                raise UserInputError(
                    f"unknown SID acquisition configuration keys: {sorted(unknown)}"
                )
            result = cls(**raw)
        if result.dataset_id != "saberzl/SID_Set":
            raise UserInputError("SID acquisition dataset_id must be saberzl/SID_Set")
        if result.split != "validation" or not result.streaming:
            raise UserInputError(
                "SID acquisition must stream the pinned validation split"
            )
        if not result.revision.strip():
            raise UserInputError("SID acquisition revision must be recorded")
        if result.per_class < 1:
            raise UserInputError("SID acquisition per_class must be at least 1")
        images_directory = _safe_relative_path(
            result.images_directory, "images_directory"
        )
        manifest_name = _safe_relative_path(result.manifest_name, "manifest_name")
        _validate_output_layout(images_directory, manifest_name)
        return replace(
            result,
            images_directory=images_directory.as_posix(),
            manifest_name=manifest_name.as_posix(),
        )


@dataclass(frozen=True)
class AcquisitionSummary:
    output_root: Path
    manifest_path: Path
    counts: dict[int, int]
    dataset_revision: str
    observed_dataset_revision: str
    licence_identifier: str
    config_sha256: str


@dataclass(frozen=True)
class SourcePolicy:
    name: str
    root: Path
    allowed_labels: tuple[int, ...]
    generator_labeled: bool = False


@dataclass(frozen=True)
class PrimaryManifestPolicy:
    sources: tuple[SourcePolicy, ...]
    maximum_corrupt_fraction: float
    require_both_labels: bool
    minimum_generator_families: int


@dataclass(frozen=True)
class _RecordAdapter:
    records: Sequence[ManifestRecord]

    def scan(self) -> Iterator[ManifestRecord]:
        yield from self.records


def select_balanced_binary_rows(
    rows: Iterable[Mapping[str, Any]], per_class: int
) -> Iterator[Mapping[str, Any]]:
    """Yield only binary rows and stop as soon as both class caps are full."""
    if per_class < 1:
        raise UserInputError("per_class must be at least 1")
    counts = {0: 0, 1: 0}
    for row in rows:
        try:
            label = int(row["label"])
        except (KeyError, TypeError, ValueError) as error:
            raise DatasetAcquisitionError("SID row has no valid label") from error
        if label not in counts or counts[label] >= per_class:
            continue
        counts[label] += 1
        yield row
        if counts == {0: per_class, 1: per_class}:
            return


def hash_acquisition_config(config: Mapping[str, Any] | SidAcquisitionConfig) -> str:
    """Hash a configuration by canonical JSON, independent of mapping insertion order."""
    payload = asdict(config) if isinstance(config, SidAcquisitionConfig) else dict(config)
    canonical = json.dumps(
        _json_value(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def acquire_sid_subset(
    config: SidAcquisitionConfig | Mapping[str, Any] | Path,
    output_root: Path,
    *,
    dataset_loader: Callable[..., Iterable[Mapping[str, Any]]] | None = None,
) -> AcquisitionSummary:
    """Stream a complete balanced SID subset into a canonical local manifest."""
    resolved = SidAcquisitionConfig.from_value(config)
    output_root = Path(output_root)
    if output_root.exists():
        raise UserInputError(f"acquisition output root already exists: {output_root}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.acquiring-", dir=output_root.parent)
    )
    published = False
    try:
        loader = dataset_loader or _load_hugging_face_dataset
        dataset = loader(
            resolved.dataset_id,
            split=resolved.split,
            streaming=True,
            revision=resolved.revision,
        )
        observed_revision = _observed_revision(dataset, resolved.revision)
        selected = list(select_balanced_binary_rows(dataset, resolved.per_class))
        counts = {label: sum(int(row[resolved.label_field]) == label for row in selected) for label in (0, 1)}
        underfilled = [label for label, count in counts.items() if count != resolved.per_class]
        if underfilled:
            details = ", ".join(
                f"label {label}: {counts[label]}/{resolved.per_class}" for label in underfilled
            )
            raise DatasetAcquisitionError(f"SID acquisition underfilled {details}")

        rows_for_adapter = _save_selected_images(selected, resolved, staging)
        records = list(SidSetAdapter(version=resolved.revision).scan_rows(rows_for_adapter))
        manifest_path = _resolved_write_path(staging, resolved.manifest_name, "manifest_name")
        result = build_manifest([_RecordAdapter(records)], manifest_path, max_corrupt_fraction=0.0)
        if result.valid_count != 2 * resolved.per_class:
            raise DatasetAcquisitionError(
                f"SID acquisition produced {result.valid_count} validated rows, "
                f"expected {2 * resolved.per_class}"
            )

        config_sha256 = hash_acquisition_config(resolved)
        metadata = {
            "config_sha256": config_sha256,
            "counts": counts,
            "dataset_id": resolved.dataset_id,
            "dataset_revision": resolved.revision,
            "licence_identifier": SID_SET.licence_identifier,
            "observed_dataset_revision": observed_revision,
            "split": resolved.split,
        }
        _atomic_write_json(staging / ACQUISITION_METADATA_NAME, metadata)
        staging.replace(output_root)
        published = True
        relocated_records = [
            record.model_copy(
                update={"path": output_root / record.path.relative_to(staging)}
            )
            for record in records
        ]
        build_manifest(
            [_RecordAdapter(relocated_records)],
            _resolved_write_path(output_root, resolved.manifest_name, "manifest_name"),
            max_corrupt_fraction=0.0,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        if published:
            shutil.rmtree(output_root, ignore_errors=True)
        raise

    return AcquisitionSummary(
        output_root=output_root,
        manifest_path=_resolved_write_path(output_root, resolved.manifest_name, "manifest_name"),
        counts=counts,
        dataset_revision=resolved.revision,
        observed_dataset_revision=observed_revision,
        licence_identifier=SID_SET.licence_identifier,
        config_sha256=config_sha256,
    )


def validate_wildfake_root(root: Path) -> Path:
    """Validate a manually obtained WildFake export and explain both official sources."""
    root = Path(root)
    if root.is_dir() and any(path.is_file() for path in root.rglob("*")):
        return root
    raise DatasetAcquisitionError(wildfake_manual_acquisition_message(root))


def load_primary_policy(path: Path) -> PrimaryManifestPolicy:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise UserInputError("primary manifest policy must be a mapping")
    sources = tuple(
        SourcePolicy(
            name=str(item["name"]),
            root=Path(item["root"]),
            allowed_labels=tuple(int(label) for label in item["allowed_labels"]),
            generator_labeled=bool(item.get("generator_labeled", False)),
        )
        for item in raw["sources"]
    )
    return PrimaryManifestPolicy(
        sources=sources,
        maximum_corrupt_fraction=float(raw["maximum_corrupt_fraction"]),
        require_both_labels=bool(raw["require_both_labels"]),
        minimum_generator_families=int(raw["minimum_generator_families"]),
    )


def validate_primary_manifest(frame: pd.DataFrame, policy: PrimaryManifestPolicy) -> None:
    """Enforce binary labels and the cross-source unseen-generator release gate."""
    required = {"label", "dataset_name", "generator_family"}
    missing = required - set(frame.columns)
    if missing:
        raise DatasetPolicyError(f"primary manifest is missing columns: {sorted(missing)}")

    dataset_names = frame["dataset_name"]
    unidentified = dataset_names.isna() | dataset_names.fillna("").astype(str).str.strip().eq("")
    if unidentified.any():
        raise DatasetPolicyError("primary manifest dataset_name values cannot be missing or blank")

    approved = {source.name: source for source in policy.sources}
    unapproved = set(frame["dataset_name"].dropna().astype(str)) - set(approved)
    if unapproved:
        raise DatasetPolicyError(f"primary manifest contains unapproved sources: {sorted(unapproved)}")
    for name, source in approved.items():
        labels = set(frame.loc[frame["dataset_name"] == name, "label"].astype(int))
        disallowed = labels - set(source.allowed_labels)
        if disallowed:
            raise DatasetPolicyError(f"source {name} contains disallowed labels: {sorted(disallowed)}")

    labels = set(frame["label"].dropna().astype(int))
    if policy.require_both_labels and labels != {0, 1}:
        raise DatasetPolicyError("primary manifest must contain both labels 0 and 1")

    generator_sources = {source.name for source in policy.sources if source.generator_labeled}
    fake = frame[(frame["label"] == 1) & frame["dataset_name"].isin(generator_sources)]
    families = {
        str(value).strip()
        for value in fake["generator_family"].dropna()
        if str(value).strip()
    }
    if len(families) < policy.minimum_generator_families:
        raise DatasetPolicyError(
            "primary manifest requires at least "
            f"{policy.minimum_generator_families} fake generator families across approved "
            f"generator-labeled sources; found {len(families)}"
        )


def _save_selected_images(
    selected: Sequence[Mapping[str, Any]], config: SidAcquisitionConfig, staging: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for row in selected:
        image_id = str(row[config.id_field])
        if image_id in seen_ids:
            raise DatasetAcquisitionError(f"SID acquisition contains duplicate img_id: {image_id}")
        seen_ids.add(image_id)
        label = int(row[config.label_field])
        image = row.get(config.image_field)
        if not isinstance(image, Image.Image):
            raise DatasetAcquisitionError(f"SID row {image_id} has no decoded PIL image")
        relative = Path(str(label)) / (
            f"{hashlib.sha256(image_id.encode('utf-8')).hexdigest()}.png"
        )
        images_root = _resolved_write_path(
            staging, config.images_directory, "images_directory"
        )
        destination = _resolved_write_path(images_root, relative, "images_directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        rgb = image.convert("RGB")
        rgb.save(destination, format="PNG")
        rows.append(
            {
                "img_id": image_id,
                "label": label,
                "image_path": destination,
                "width": rgb.width,
                "height": rgb.height,
                "file_format": "PNG",
            }
        )
    return rows


def _observed_revision(dataset: object, fallback: str) -> str:
    for attribute in ("dataset_revision", "revision", "_revision"):
        value = getattr(dataset, attribute, None)
        if value:
            return str(value)
    info = getattr(dataset, "info", None)
    version = getattr(info, "version", None)
    return str(version) if version else fallback


def _load_hugging_face_dataset(dataset_id: str, **kwargs: object):
    from datasets import load_dataset

    return load_dataset(dataset_id, **kwargs)


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _safe_relative_path(value: object, field: str) -> Path:
    if not isinstance(value, str):
        raise UserInputError(f"{field} must be a string containing a relative path")
    text = value.strip()
    if not text:
        raise UserInputError(f"{field} must be a non-empty relative path")
    try:
        posix = PurePosixPath(text)
        windows = PureWindowsPath(text)
    except (OSError, ValueError) as error:
        raise UserInputError(f"{field} must be a safe relative path") from error
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(posix.anchor)
        or bool(windows.anchor)
        or bool(windows.drive)
        or ".." in posix.parts
        or ".." in windows.parts
        or text in {".", "./", ".\\"}
    ):
        raise UserInputError(f"{field} must be a safe relative path beneath output_root")
    return Path(*windows.parts)


def _validate_output_layout(images_directory: Path, manifest_name: Path) -> None:
    outputs = (
        ("images_directory", images_directory),
        ("manifest_name", manifest_name),
        ("acquisition metadata", Path(ACQUISITION_METADATA_NAME)),
    )
    for index, (left_name, left_path) in enumerate(outputs):
        for right_name, right_path in outputs[index + 1 :]:
            if _relative_paths_overlap(left_path, right_path):
                raise UserInputError(
                    "acquisition output paths overlap: "
                    f"{left_name}={left_path.as_posix()} and "
                    f"{right_name}={right_path.as_posix()}"
                )


def _relative_paths_overlap(first: Path, second: Path) -> bool:
    first_parts = tuple(part.casefold() for part in first.parts)
    second_parts = tuple(part.casefold() for part in second.parts)
    common_length = min(len(first_parts), len(second_parts))
    return first_parts[:common_length] == second_parts[:common_length]


def _resolved_write_path(root: Path, relative: str | Path, field: str) -> Path:
    root = root.resolve()
    destination = (root / relative).resolve()
    try:
        destination.relative_to(root)
    except ValueError as error:
        raise UserInputError(f"{field} resolves outside the intended output root") from error
    return destination


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
