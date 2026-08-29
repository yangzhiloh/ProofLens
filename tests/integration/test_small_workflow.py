import subprocess
import sys
from pathlib import Path


def test_cli_lists_required_commands() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "prooflens.cli", "--help"],
        text=True, capture_output=True, check=True,
    )
    for command in ("acquire", "manifest", "audit", "split", "train", "evaluate", "select", "calibrate", "report", "export", "app"):
        assert command in result.stdout


def test_small_reproduction_creates_required_artifacts(tmp_path: Path) -> None:
    from scripts.reproduce_small import reproduce_small

    result = reproduce_small(tmp_path)
    assert result.checkpoint.exists()
    assert result.predictions.exists()
    assert result.metrics.exists()
    assert result.robustness_markdown.exists()
