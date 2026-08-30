from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from scripts.task8_preflight import run_preflight, write_pilot_manifest


ROOT = Path(__file__).resolve().parents[2]


def test_preflight_is_read_only_and_keeps_unverified_sources_blocked() -> None:
    report = run_preflight(ROOT)
    checks = {check["name"]: check for check in report["checks"]}

    assert report["safety"] == "No dataset download or training was performed."
    assert checks["licence-sid_set"]["status"] == "pass"
    assert checks["licence-cifake"]["status"] == "block"
    assert checks["licence-aigenimages2026"]["status"] == "pass"
    assert checks["robustness-transforms"]["status"] == "pass"
    assert report["ready_for_training"] is False


def test_source_registry_keeps_cifake_out_of_primary_training() -> None:
    policy = yaml.safe_load(
        (ROOT / "configs/data/task8_sources.yaml").read_text(encoding="utf-8")
    )

    assert policy["cifake_primary_training"] is False
    assert policy["minimum_generator_families"] == 3


def test_pilot_manifest_is_deterministic_balanced_and_bounded(tmp_path: Path) -> None:
    rows = []
    for split in ("train", "validation", "test"):
        for label in (0, 1):
            for index in range(5):
                rows.append(
                    {
                        "sample_id": f"{split}-{label}-{4 - index}",
                        "split": split,
                        "label": label,
                        "generator_family": "authentic" if label == 0 else "generator-a",
                    }
                )
    source = tmp_path / "source.parquet"
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    pd.DataFrame(rows).to_parquet(source, index=False)

    write_pilot_manifest(source, first, per_label_split=2)
    write_pilot_manifest(source, second, per_label_split=2)
    first_frame = pd.read_parquet(first)

    assert first.read_bytes() == second.read_bytes()
    assert first_frame.groupby(["split", "label"]).size().eq(2).all()
    assert len(first_frame) == 12


def test_pilot_experiment_uses_isolated_small_run() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/experiments/e0_pilot.yaml").read_text(encoding="utf-8")
    )

    assert config["training"]["epochs"] == 1
    assert config["training"]["batch_size"] == 4
    assert config["model"]["stage"] == "head"
    assert config["output_dir"] == "artifacts/runs/pilot-e0"
