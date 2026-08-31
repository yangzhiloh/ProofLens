from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_platform_runners_share_locked_commands_and_artifact_layout() -> None:
    powershell = (ROOT / "scripts" / "prooflens.ps1").read_text(encoding="utf-8")
    shell = (ROOT / "scripts" / "prooflens.sh").read_text(encoding="utf-8")

    for runner in (powershell, shell):
        assert "sync --locked --extra dev" in runner
        assert "run --locked --extra dev python -m pytest -q" in runner
        assert "task8_preflight.py" in runner
        assert "--publish-demo-artifacts" in runner
        assert "export/artifact_manifest.json" in runner
        assert "export/model.onnx" in runner
        assert "export/calibration.json" in runner


def test_ci_executes_both_platform_runners() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert ".\\scripts\\prooflens.ps1 artifacts" in workflow
    assert "bash scripts/prooflens.sh artifacts" in workflow
    assert "'model.onnx', 'calibration.json', 'artifact_manifest.json'" in workflow
