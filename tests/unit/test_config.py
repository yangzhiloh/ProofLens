from pathlib import Path

import pytest

from prooflens.config import ExperimentConfig, load_config


def test_load_config_rejects_unknown_keys(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("model:\n  name: facebook/dinov2-base\nunknown: true\n")
    with pytest.raises(ValueError):
        load_config(path)


def test_resolve_makes_manifest_and_output_absolute(tmp_path: Path) -> None:
    config = ExperimentConfig.model_validate({
        "seed": 17,
        "data": {"manifest": "artifacts/manifests/train.parquet"},
        "model": {"name": "facebook/dinov2-base", "stage": "head"},
        "training": {"epochs": 1, "batch_size": 2},
        "output_dir": "artifacts/runs/e0",
    })
    resolved = config.resolve(tmp_path)
    assert resolved.data.manifest == tmp_path / "artifacts/manifests/train.parquet"
    assert resolved.output_dir == tmp_path / "artifacts/runs/e0"
