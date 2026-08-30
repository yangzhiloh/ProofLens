from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import imagehash
import pandas as pd
import pyarrow as pa
import pytest
import yaml
from PIL import Image, ImageOps
from pydantic import ValidationError

from prooflens.data.hashing import sha256_file
from prooflens.data.splitting import build_split_groups


def _write_image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), color).save(path)


def write_primary_fixture(tmp_path: Path) -> Path:
    sid_root = tmp_path / "sid_set"
    sid_real = sid_root / "images" / "real.png"
    sid_fake = sid_root / "images" / "fake.png"
    _write_image(sid_real, "white")
    _write_image(sid_fake, "black")
    pd.DataFrame(
        [
            {
                "sample_id": "sid-real",
                "path": str(sid_real),
                "label": 0,
                "dataset_name": "sid_set",
                "dataset_version": "pinned-revision",
                "generator_family": "authentic",
                "source_group_id": "sid-real",
                "original_image_id": "sid-real",
                "width": 4,
                "height": 3,
                "file_format": "PNG",
                "licence_identifier": "CC-BY-4.0",
                "content_checksum": "",
                "perceptual_hash": "",
                "split": "unassigned",
            },
            {
                "sample_id": "sid-fake",
                "path": str(sid_fake),
                "label": 1,
                "dataset_name": "sid_set",
                "dataset_version": "pinned-revision",
                "generator_family": "generated",
                "source_group_id": "sid-fake",
                "original_image_id": "sid-fake",
                "width": 4,
                "height": 3,
                "file_format": "PNG",
                "licence_identifier": "CC-BY-4.0",
                "content_checksum": "",
                "perceptual_hash": "",
                "split": "unassigned",
            },
        ]
    ).to_parquet(sid_root / "manifest.parquet", index=False)
    wildfake_root = tmp_path / "wildfake"
    _write_image(wildfake_root / "real" / "camera" / "real.png", "gray")
    for family, color in (("flux", "red"), ("sdxl", "blue"), ("dalle3", "green")):
        _write_image(wildfake_root / "fake" / family / "fake.png", color)
    config_path = tmp_path / "primary.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {"name": "sid_set", "root": str(sid_root), "allowed_labels": [0, 1]},
                    {
                        "name": "wildfake",
                        "root": str(wildfake_root),
                        "allowed_labels": [0, 1],
                        "generator_labeled": True,
                    },
                ],
                "maximum_corrupt_fraction": 0.01,
                "require_both_labels": True,
                "minimum_generator_families": 3,
            }
        ),
        encoding="utf-8",
    )
    return config_path


def test_primary_manifest_cli_combines_acquired_sid_and_wildfake(tmp_path: Path) -> None:
    from prooflens.cli import run_manifest_cli

    output = tmp_path / "primary.parquet"

    exit_code = run_manifest_cli(
        argparse.Namespace(config=write_primary_fixture(tmp_path), output=output)
    )

    frame = pd.read_parquet(output)
    assert exit_code == 0
    assert set(frame["dataset_name"]) == {"sid_set", "wildfake"}
    assert frame.loc[frame["label"] == 1, "generator_family"].nunique() >= 3


def test_primary_manifest_cli_enriches_hashes_for_split_contract(tmp_path: Path) -> None:
    from prooflens.cli import run_manifest_cli

    config = write_primary_fixture(tmp_path)
    acquired_sid = pd.read_parquet(tmp_path / "sid_set" / "manifest.parquet")
    output = tmp_path / "primary.parquet"

    exit_code = run_manifest_cli(argparse.Namespace(config=config, output=output))

    frame = pd.read_parquet(output)
    expected_sha256: list[str] = []
    expected_phash: list[str] = []
    for value in frame["path"]:
        path = Path(value)
        expected_sha256.append(hashlib.sha256(path.read_bytes()).hexdigest())
        with Image.open(path) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            expected_phash.append(str(imagehash.phash(normalized)))
    assert exit_code == 0
    assert frame["content_checksum"].tolist() == expected_sha256
    assert frame["perceptual_hash"].tolist() == expected_phash
    acquired_identity = acquired_sid.set_index("sample_id")[["path", "label"]].sort_index()
    primary_sid_identity = (
        frame.loc[frame["dataset_name"] == "sid_set"]
        .set_index("sample_id")[["path", "label"]]
        .sort_index()
    )
    pd.testing.assert_frame_equal(primary_sid_identity, acquired_identity)
    grouped = build_split_groups(frame, max_phash_distance=4)
    assert len(grouped) == len(frame)
    assert grouped["split_group_id"].str.startswith("split-").all()


def test_manifest_cli_reports_malformed_acquired_manifest_as_data_integrity_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from prooflens.cli import dispatch

    config_path = write_primary_fixture(tmp_path)
    pd.DataFrame([{"sample_id": "sid-real"}]).to_parquet(
        tmp_path / "sid_set" / "manifest.parquet", index=False
    )

    exit_code = dispatch(
        argparse.Namespace(command="manifest", config=config_path, output=tmp_path / "primary.parquet")
    )

    assert exit_code == 3
    assert "data integrity error" in capsys.readouterr().err


def test_select_command_uses_current_checkpoint_ranking(tmp_path) -> None:
    from prooflens.cli import run_select_cli

    first = tmp_path / "run-a"
    second = tmp_path / "run-b"
    first.mkdir()
    second.mkdir()
    for run, clean, robust, worst in (
        (first, 0.90, 0.70, 0.60),
        (second, 0.85, 0.80, 0.72),
    ):
        (run / "metrics.json").write_text(
            json.dumps(
                {
                    "ranking": {
                        "clean_auc": clean,
                        "macro_robust_auc": robust,
                        "worst_family_auc": worst,
                        "unseen_generator_auc": 0.75,
                        "model_parameters": 10,
                    }
                }
            ),
            encoding="utf-8",
        )
        (run / "run_metadata.json").write_text(
            json.dumps({"split_sha256": ("a" if run == first else "b") * 64}),
            encoding="utf-8",
        )
    output = tmp_path / "selection.json"

    exit_code = run_select_cli(argparse.Namespace(runs=[first, second], output=output))

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["checkpoint_id"] == "run-b"
    assert payload["run_dir"] == str(second)
    assert payload["validation_split_hash"] == "b" * 64


def test_train_command_persists_resolved_config(tmp_path, monkeypatch) -> None:
    from prooflens.cli import run_train_cli
    from prooflens.training import trainer

    manifest = tmp_path / "split.parquet"
    manifest.write_bytes(b"fixture")
    output_dir = tmp_path / "run"
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 17,
                "data": {"manifest": str(manifest)},
                "model": {"name": "fixture/model", "stage": "head"},
                "training": {"epochs": 1, "batch_size": 2},
                "output_dir": str(output_dir),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        trainer,
        "run_training",
        lambda config: SimpleNamespace(best_checkpoint=config.output_dir / "best.pt"),
    )

    exit_code = run_train_cli(
        argparse.Namespace(
            config=config_path,
            config_from_selection=None,
            seed=None,
            output=None,
        )
    )

    saved = yaml.safe_load((output_dir / "config.yaml").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert Path(saved["data"]["manifest"]) == manifest
    assert Path(saved["output_dir"]) == output_dir


def test_calibrate_command_writes_current_artifact_schema(tmp_path) -> None:
    from prooflens.cli import run_calibrate_cli

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    split = tmp_path / "split.parquet"
    split.write_bytes(b"fixture split")
    split_hash = sha256_file(split)
    split.with_suffix(".json").write_text(
        json.dumps({"split_sha256": split_hash}),
        encoding="utf-8",
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"split_sha256": split_hash}),
        encoding="utf-8",
    )
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "seed": 17,
                "data": {"manifest": str(split)},
                "model": {"name": "fixture/model", "stage": "head"},
                "training": {"epochs": 1, "batch_size": 2},
                "output_dir": str(run_dir),
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "split": ["validation"] * 4,
            "condition_id": ["clean"] * 4,
            "label": [0, 0, 1, 1],
            "logit": [-2.0, -1.0, 1.0, 2.0],
        }
    ).to_parquet(run_dir / "predictions-validation.parquet", index=False)
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "validation_split_hash": split_hash,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "calibration.json"

    exit_code = run_calibrate_cli(
        argparse.Namespace(selection=selection, split="validation", output=output)
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert payload["temperature"] > 0.0
    assert 0.0 <= payload["threshold"] <= 1.0
    assert payload["validation_split_hash"] == split_hash
    updated_selection = json.loads(selection.read_text(encoding="utf-8"))
    assert updated_selection["calibration_path"] == str(output)


def test_calibrate_command_rejects_stale_split_provenance(tmp_path) -> None:
    from prooflens.cli import run_calibrate_cli
    from prooflens.errors import UserInputError

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"split_sha256": "a" * 64}),
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "validation_split_hash": "b" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(UserInputError, match="does not match"):
        run_calibrate_cli(
            argparse.Namespace(
                selection=selection,
                split="validation",
                output=tmp_path / "calibration.json",
            )
        )


def test_app_parser_requires_calibration() -> None:
    from prooflens.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["app", "--backend", "onnx", "--model", "model.onnx"])


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("missing selection"),
        json.JSONDecodeError("invalid selection", "{", 1),
        KeyError("run_dir"),
    ],
)
def test_dispatch_normalizes_invalid_input_files(error, monkeypatch, capsys) -> None:
    from prooflens import cli

    def fail(args):
        raise error

    monkeypatch.setitem(cli.COMMAND_HANDLERS, "select", fail)

    exit_code = cli.dispatch(argparse.Namespace(command="select"))

    assert exit_code == 2
    assert "configuration error" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("error", "expected_code", "message"),
    [
        (yaml.YAMLError("invalid yaml"), 2, "configuration error"),
        (pa.ArrowInvalid("invalid parquet"), 3, "data integrity error"),
    ],
)
def test_dispatch_normalizes_config_and_parquet_errors(
    error, expected_code, message, monkeypatch, capsys
) -> None:
    from prooflens import cli

    monkeypatch.setitem(cli.COMMAND_HANDLERS, "select", lambda args: (_ for _ in ()).throw(error))

    assert cli.dispatch(argparse.Namespace(command="select")) == expected_code
    assert message in capsys.readouterr().err


def test_dispatch_normalizes_pydantic_config_errors(monkeypatch, capsys) -> None:
    from prooflens import cli
    from prooflens.config import ExperimentConfig

    with pytest.raises(ValidationError) as captured:
        ExperimentConfig.model_validate({})
    monkeypatch.setitem(
        cli.COMMAND_HANDLERS,
        "train",
        lambda args: (_ for _ in ()).throw(captured.value),
    )

    assert cli.dispatch(argparse.Namespace(command="train")) == 2
    assert "configuration error" in capsys.readouterr().err


def test_report_command_applies_frozen_calibration(tmp_path) -> None:
    from prooflens.cli import run_report_cli
    from prooflens.data.transforms import canonical_specs

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rows: list[dict[str, object]] = []
    labels = (0, 0, 1, 1)
    logits = (-2.0, -1.0, 1.0, 2.0)
    for condition_id, family, split in [
        ("clean", "clean", "test"),
        *((spec.condition_id, spec.family, "test") for spec in canonical_specs()),
        ("clean", "clean", "generator_test"),
    ]:
        for index, (label, logit) in enumerate(zip(labels, logits, strict=True)):
            rows.append(
                {
                    "sample_id": f"{split}-{condition_id}-{index}",
                    "label": label,
                    "logit": logit,
                    "score": 1.0 / (1.0 + math.exp(-logit)),
                    "split": split,
                    "generator_family": "real" if label == 0 else "fixture-ai",
                    "transform_family": family,
                    "condition_id": condition_id,
                    "checkpoint_id": "best",
                }
            )
    pd.DataFrame(rows).to_parquet(run_dir / "predictions-test.parquet", index=False)
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "temperature": 1.0,
                "threshold": 0.5,
                "validation_split_hash": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "calibration_path": str(calibration),
                "validation_split_hash": "a" * 64,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "report"

    exit_code = run_report_cli(argparse.Namespace(selection=selection, output=output))

    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert metrics["operating_point"]["threshold"] == 0.5
    assert metrics["operating_point"]["accuracy"] == 1.0


def test_verified_onnx_is_not_published_when_parity_fails(tmp_path) -> None:
    from prooflens.cli import _publish_verified_onnx

    output = tmp_path / "model.onnx"

    def export_fn(model, sample, destination):
        destination.write_bytes(b"unverified")
        return destination

    def verify_fn(model, path, sample, **kwargs):
        raise RuntimeError("parity failed")

    with pytest.raises(RuntimeError, match="parity failed"):
        _publish_verified_onnx(
            object(),
            object(),
            output,
            temperature=1.0,
            export_fn=export_fn,
            verify_fn=verify_fn,
        )

    assert not output.exists()
    assert not list(tmp_path.glob(".*.unverified-*.onnx"))


def test_app_command_uses_checkpoint_factory_and_launches(tmp_path, monkeypatch) -> None:
    from prooflens.cli import run_app_cli
    from prooflens.inference import preprocess, torch_backend
    from prooflens.web import app as web_app

    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"fixture")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps({"temperature": 1.0, "threshold": 0.5}),
        encoding="utf-8",
    )

    class FakeBackend:
        model_version = "fixture"
        preprocessing_version = "fixture-224"

        def predict_logit(self, image) -> float:
            return 0.0

    class FakeApp:
        launched = False

        def launch(self) -> None:
            self.launched = True

    fake_app = FakeApp()
    monkeypatch.setattr(
        torch_backend.TorchLogitBackend,
        "from_checkpoint",
        lambda *args, **kwargs: FakeBackend(),
    )
    monkeypatch.setattr(preprocess, "create_dinov2_processor", lambda: object())
    monkeypatch.setattr(web_app, "create_app", lambda service: fake_app)

    exit_code = run_app_cli(
        argparse.Namespace(
            backend="torch",
            checkpoint=checkpoint,
            model=None,
            calibration=calibration,
        )
    )

    assert exit_code == 0
    assert fake_app.launched


def test_evaluate_stress_command_requires_selection_split_and_output() -> None:
    from prooflens.cli import COMMANDS, build_parser

    parsed = build_parser().parse_args(
        [
            "evaluate-stress",
            "--selection",
            "selection.json",
            "--split",
            "test",
            "--output",
            "stress-output",
        ]
    )

    assert "evaluate-stress" in COMMANDS
    assert parsed.command == "evaluate-stress"
    assert parsed.selection == Path("selection.json")
    assert parsed.split == "test"
    assert parsed.output == Path("stress-output")


def test_evaluate_stress_uses_selected_checkpoint_and_writes_split_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from prooflens.cli import run_evaluate_stress_cli
    from prooflens.data.hashing import sha256_file
    from prooflens.inference import preprocess, torch_backend

    run_dir = tmp_path / "selected-run"
    run_dir.mkdir()
    images = []
    for split, label, color in (
        ("validation", 0, "red"),
        ("validation", 1, "blue"),
        ("test", 0, "green"),
        ("test", 1, "yellow"),
    ):
        path = tmp_path / f"{split}-{label}.png"
        _write_image(path, color)
        images.append(
            {
                "sample_id": f"{split}-{label}",
                "path": str(path),
                "label": label,
                "dataset_name": "fixture",
                "dataset_version": "fixture-v1",
                "generator_family": "real" if label == 0 else "generator",
                "source_group_id": f"group-{split}-{label}",
                "original_image_id": f"image-{split}-{label}",
                "width": 4,
                "height": 3,
                "file_format": "PNG",
                "licence_identifier": "CC0-1.0",
                "content_checksum": "",
                "perceptual_hash": "",
                "split": split,
                "split_group_id": f"group-{split}-{label}",
            }
        )
    manifest = tmp_path / "split.parquet"
    pd.DataFrame(images).to_parquet(manifest, index=False)
    split_hash = sha256_file(manifest)
    manifest.with_suffix(".json").write_text(
        json.dumps({"split_sha256": split_hash}), encoding="utf-8"
    )
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"split_sha256": split_hash}), encoding="utf-8"
    )
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "seed": 17,
                "data": {"manifest": str(manifest)},
                "model": {"name": "fixture/model", "stage": "head"},
                "training": {"epochs": 1, "batch_size": 2},
                "output_dir": str(run_dir),
            }
        ),
        encoding="utf-8",
    )
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "checkpoint_id": "selected-run",
                "run_dir": str(run_dir),
                "validation_split_hash": split_hash,
            }
        ),
        encoding="utf-8",
    )
    requested_checkpoints: list[Path] = []

    class FakeBackend:
        def predict_logit(self, image: Image.Image) -> float:
            return float(sum(image.getpixel((0, 0)))) / 1000

    def from_checkpoint(path: Path, **_kwargs: object) -> FakeBackend:
        requested_checkpoints.append(path)
        return FakeBackend()

    monkeypatch.setattr(torch_backend.TorchLogitBackend, "from_checkpoint", from_checkpoint)
    monkeypatch.setattr(preprocess, "create_dinov2_processor", lambda: object())

    output = tmp_path / "stress-output"
    exit_code = run_evaluate_stress_cli(
        argparse.Namespace(selection=selection, split="test", output=output)
    )

    predictions = pd.read_parquet(output / "predictions-stress.parquet")
    metrics = json.loads((output / "stress-metrics.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert requested_checkpoints == [run_dir / "checkpoints" / "best.pt"]
    assert set(predictions["split"]) == {"test"}
    assert len(predictions) == 8
    assert set(metrics["conditions"]) == {
        "webp_q80",
        "webp_q50",
        "screenshot_1440",
        "screenshot_1080",
    }
