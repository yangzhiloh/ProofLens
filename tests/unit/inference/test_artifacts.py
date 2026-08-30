from __future__ import annotations

import json
from pathlib import Path

import pytest

from prooflens.data.hashing import sha256_file
from prooflens.errors import DataIntegrityError
from prooflens.inference.artifacts import (
    discover_artifact_manifest,
    load_artifact_metadata,
    validate_artifact_bundle,
    validate_artifact_pair,
    write_artifact_manifest,
)


def _write_bundle(root: Path) -> tuple[Path, Path, Path]:
    export = root / "export"
    export.mkdir(parents=True)
    model = export / "model.onnx"
    calibration = export / "calibration.json"
    selection = root / "selection.json"
    model.write_bytes(b"fixture model")
    calibration.write_text('{"temperature": 1.0, "threshold": 0.5}\n', encoding="utf-8")
    selection.write_text('{"checkpoint_id": "fixture"}\n', encoding="utf-8")
    manifest = write_artifact_manifest(
        export / "artifact_manifest.json",
        artifact_tier="deterministic-fixture-demo",
        model_version="prooflens-fixture-onnx",
        preprocessing_name="fixture",
        preprocessing_version="fixture-rgb-224-v1",
        files={"model": model, "calibration": calibration, "selection": selection},
    )
    return model, calibration, manifest


def test_artifact_manifest_binds_paths_hashes_and_preprocessing(tmp_path: Path) -> None:
    model, calibration, manifest = _write_bundle(tmp_path)

    metadata = load_artifact_metadata(manifest)
    validate_artifact_pair(metadata, model_path=model, calibration_path=calibration)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["preprocessing"] == {
        "name": "fixture",
        "version": "fixture-rgb-224-v1",
    }
    assert payload["files"]["model"]["sha256"] == sha256_file(model)
    assert discover_artifact_manifest(model) == manifest


def test_artifact_manifest_rejects_tampered_model(tmp_path: Path) -> None:
    model, calibration, manifest = _write_bundle(tmp_path)
    model.write_bytes(b"tampered")

    with pytest.raises(DataIntegrityError, match="hash validation"):
        validate_artifact_pair(
            load_artifact_metadata(manifest),
            model_path=model,
            calibration_path=calibration,
        )


def test_artifact_manifest_rejects_mixed_calibration(tmp_path: Path) -> None:
    model, _, manifest = _write_bundle(tmp_path / "first")
    _, other_calibration, _ = _write_bundle(tmp_path / "second")

    with pytest.raises(DataIntegrityError, match="does not match"):
        validate_artifact_pair(
            load_artifact_metadata(manifest),
            model_path=model,
            calibration_path=other_calibration,
        )


def test_artifact_manifest_rejects_unknown_preprocessing(tmp_path: Path) -> None:
    _, _, manifest = _write_bundle(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["preprocessing"]["name"] = "unknown"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataIntegrityError, match="unsupported artifact preprocessing"):
        load_artifact_metadata(manifest)


@pytest.mark.parametrize("schema_version", [0, 2, "1", None])
def test_artifact_manifest_rejects_unsupported_schema_versions(
    tmp_path: Path, schema_version: object
) -> None:
    _, _, manifest = _write_bundle(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["schema_version"] = schema_version
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataIntegrityError, match="unsupported schema version"):
        load_artifact_metadata(manifest)


@pytest.mark.parametrize("relative", ["../../model.onnx", "/tmp/model.onnx", "C:/model.onnx"])
def test_artifact_manifest_rejects_unsafe_file_paths(tmp_path: Path, relative: str) -> None:
    _, _, manifest = _write_bundle(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"]["model"]["path"] = relative
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataIntegrityError, match="model path is invalid"):
        load_artifact_metadata(manifest)


@pytest.mark.parametrize("digest", ["0" * 63, "G" * 64, 123])
def test_artifact_manifest_rejects_malformed_file_hashes(tmp_path: Path, digest: object) -> None:
    _, _, manifest = _write_bundle(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"]["selection"]["sha256"] = digest
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataIntegrityError, match="selection hash is invalid"):
        load_artifact_metadata(manifest)


def test_artifact_manifest_validates_every_file_entry_on_load(tmp_path: Path) -> None:
    _, _, manifest = _write_bundle(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["files"]["selection"]["unexpected"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataIntegrityError, match="contain only path and sha256"):
        load_artifact_metadata(manifest)


def test_artifact_manifest_rejects_unknown_schema_fields(tmp_path: Path) -> None:
    _, _, manifest = _write_bundle(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["future_field"] = True
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(DataIntegrityError, match="fields do not match schema version 1"):
        load_artifact_metadata(manifest)


def test_artifact_bundle_verifies_ancillary_files(tmp_path: Path) -> None:
    _, _, manifest = _write_bundle(tmp_path)
    (tmp_path / "selection.json").write_text("tampered\n", encoding="utf-8")

    with pytest.raises(DataIntegrityError, match="selection failed artifact hash validation"):
        validate_artifact_bundle(load_artifact_metadata(manifest))
