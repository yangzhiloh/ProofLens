from __future__ import annotations

import json
from pathlib import Path

import pytest

from prooflens.data.hashing import sha256_file
from prooflens.errors import DataIntegrityError
from prooflens.inference.artifacts import (
    discover_artifact_manifest,
    load_artifact_metadata,
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
