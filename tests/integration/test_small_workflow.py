from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch


def test_cli_lists_required_commands() -> None:
    project_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(project_root / "src")
    result = subprocess.run(
        [sys.executable, "-m", "prooflens.cli", "--help"],
        text=True,
        capture_output=True,
        check=True,
        cwd=project_root,
        env=environment,
    )

    for command in (
        "acquire",
        "manifest",
        "audit",
        "split",
        "train",
        "evaluate",
        "select",
        "calibrate",
        "report",
        "export",
        "app",
    ):
        assert command in result.stdout


def test_small_reproduction_creates_required_artifacts(tmp_path: Path) -> None:
    from scripts.reproduce_small import reproduce_small

    result = reproduce_small(tmp_path)

    assert result.checkpoint.is_file()
    assert result.predictions.is_file()
    assert result.metrics.is_file()
    assert result.robustness_markdown.is_file()
    split_metadata = json.loads((tmp_path / "split.json").read_text(encoding="utf-8"))
    assert split_metadata["policy"]["max_phash_distance"] == 4
    checkpoint = torch.load(result.checkpoint, map_location="cpu", weights_only=True)
    assert any(name.startswith("backbone.") for name in checkpoint["model"])
    predictions = pd.read_parquet(result.predictions)
    assert not predictions.duplicated(["sample_id", "split", "condition_id", "checkpoint_id"]).any()
    clean = predictions[
        (predictions["split"] == "validation") & (predictions["condition_id"] == "clean")
    ]
    assert not clean.duplicated(["sample_id", "condition_id"]).any()
