from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from prooflens.data.hashing import sha256_file
from prooflens.errors import DataIntegrityError

ARTIFACT_MANIFEST_NAME = "artifact_manifest.json"
ARTIFACT_SCHEMA_VERSION = 1
SUPPORTED_PREPROCESSING = frozenset({"dinov2", "fixture"})


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    path: Path
    artifact_tier: str
    model_version: str
    preprocessing_name: str
    preprocessing_version: str
    files: Mapping[str, Mapping[str, str]]


def write_artifact_manifest(
    path: Path,
    *,
    artifact_tier: str,
    model_version: str,
    preprocessing_name: str,
    preprocessing_version: str,
    files: Mapping[str, Path],
) -> Path:
    """Write a portable, hashed description of a published inference bundle."""

    destination = Path(path)
    required = {"model", "calibration"}
    missing = required - set(files)
    if missing:
        raise DataIntegrityError(
            "artifact manifest is missing required files: " + ", ".join(sorted(missing))
        )
    name = _preprocessing_name(preprocessing_name)
    tier = _nonempty_text(artifact_tier, "artifact tier")
    version = _nonempty_text(model_version, "model version")
    preprocessing = _nonempty_text(preprocessing_version, "preprocessing version")
    encoded_files: dict[str, dict[str, str]] = {}
    for key, value in sorted(files.items()):
        file_path = Path(value)
        if not file_path.is_file():
            raise DataIntegrityError(f"artifact manifest file does not exist: {file_path}")
        relative = Path(os.path.relpath(file_path, destination.parent)).as_posix()
        encoded_files[str(key)] = {
            "path": relative,
            "sha256": sha256_file(file_path),
        }
    payload = {
        "artifact_tier": tier,
        "files": encoded_files,
        "model_version": version,
        "preprocessing": {"name": name, "version": preprocessing},
        "schema_version": ARTIFACT_SCHEMA_VERSION,
    }
    _atomic_write(
        destination,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    return destination


def discover_artifact_manifest(model_path: Path) -> Path | None:
    """Find a sidecar beside the model or at the root of the standard artifact layout."""

    model = Path(model_path)
    candidates = (
        model.parent / ARTIFACT_MANIFEST_NAME,
        model.parent.parent / ARTIFACT_MANIFEST_NAME,
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def load_artifact_metadata(path: Path) -> ArtifactMetadata:
    """Load and schema-check preprocessing metadata without trusting its file entries."""

    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DataIntegrityError(f"artifact manifest is unreadable: {source}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise DataIntegrityError("artifact manifest has an unsupported schema version")
    preprocessing = payload.get("preprocessing")
    files = payload.get("files")
    if not isinstance(preprocessing, dict) or not isinstance(files, dict):
        raise DataIntegrityError("artifact manifest is missing preprocessing or files")
    if not all(isinstance(value, dict) for value in files.values()):
        raise DataIntegrityError("artifact manifest file entries must be objects")
    return ArtifactMetadata(
        path=source,
        artifact_tier=_nonempty_text(payload.get("artifact_tier"), "artifact tier"),
        model_version=_nonempty_text(payload.get("model_version"), "model version"),
        preprocessing_name=_preprocessing_name(preprocessing.get("name")),
        preprocessing_version=_nonempty_text(
            preprocessing.get("version"), "preprocessing version"
        ),
        files=files,
    )


def validate_artifact_pair(
    metadata: ArtifactMetadata,
    *,
    model_path: Path,
    calibration_path: Path,
) -> None:
    """Require the selected model and calibration to match the sidecar paths and hashes."""

    supplied = {"model": Path(model_path), "calibration": Path(calibration_path)}
    for name, supplied_path in supplied.items():
        entry = metadata.files.get(name)
        if not isinstance(entry, Mapping):
            raise DataIntegrityError(f"artifact manifest has no {name} entry")
        relative = entry.get("path")
        expected_hash = entry.get("sha256")
        if not isinstance(relative, str) or not relative.strip():
            raise DataIntegrityError(f"artifact manifest {name} path is invalid")
        expected_path = (metadata.path.parent / relative).resolve()
        if supplied_path.resolve() != expected_path:
            raise DataIntegrityError(
                f"selected {name} does not match artifact manifest: {supplied_path}"
            )
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise DataIntegrityError(f"artifact manifest {name} hash is invalid")
        if sha256_file(supplied_path) != expected_hash:
            raise DataIntegrityError(f"selected {name} failed artifact hash validation")


def _preprocessing_name(value: object) -> str:
    name = _nonempty_text(value, "preprocessing name")
    if name not in SUPPORTED_PREPROCESSING:
        raise DataIntegrityError(f"unsupported artifact preprocessing: {name}")
    return name


def _nonempty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataIntegrityError(f"artifact manifest {field} must be nonempty text")
    return value.strip()


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
