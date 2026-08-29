from __future__ import annotations

import hashlib
import json

from prooflens.config import ExperimentConfig


def test_metadata_distinguishes_source_manifest_from_split_artifact(tmp_path) -> None:
    from prooflens.training.run_metadata import collect_run_metadata

    split = tmp_path / "assigned.parquet"
    split.write_bytes(b"split bytes")
    source_hash = hashlib.sha256(b"source bytes").hexdigest()
    split_hash = hashlib.sha256(b"split bytes").hexdigest()
    split.with_suffix(".json").write_text(
        json.dumps(
            {
                "source_manifest_sha256": source_hash,
                "split_sha256": split_hash,
            }
        ),
        encoding="utf-8",
    )
    config = ExperimentConfig.model_validate(
        {
            "seed": 17,
            "data": {"manifest": split},
            "model": {"name": "fixture/dino", "stage": "head"},
            "training": {"epochs": 1, "batch_size": 2},
            "output_dir": tmp_path / "run",
        }
    )

    metadata = collect_run_metadata(config, device="cpu")

    assert metadata.manifest_sha256 == source_hash
    assert metadata.split_sha256 == split_hash
    assert metadata.config_sha256 != source_hash
    assert metadata.seed == 17
    assert metadata.backbone == "fixture/dino"
