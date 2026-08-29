from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REQUIRED_DOCUMENTS = (
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "docs/datasets.md",
    "docs/model-card.md",
    "docs/devpost-draft.md",
    "docs/video-script.md",
)
MODEL_CARD_FIELDS = ("intended use", "limitations", "datasets", "metrics", "ethical")


def _write_valid_project(root: Path) -> Path:
    contents = {
        "README.md": "# ProofLens\n",
        "LICENSE": "MIT License\n",
        "THIRD_PARTY_NOTICES.md": (
            "DINOv2 Apache-2.0\nSID-Set CC-BY-4.0\nCIFAKE MIT\n"
            "WildFake REQUIRES-VERIFICATION\n"
        ),
        "docs/datasets.md": "# Datasets\n",
        "docs/model-card.md": "".join(f"## {field.title()}\nDocumented.\n" for field in MODEL_CARD_FIELDS),
        "docs/devpost-draft.md": "# Devpost draft\n",
        "docs/video-script.md": "# Video script\n",
    }
    for relative, text in contents.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _run(root: Path):
    from scripts.release_check import run_release_check

    return run_release_check(root)


@pytest.mark.parametrize("relative", REQUIRED_DOCUMENTS)
def test_release_check_rejects_each_missing_required_document(
    tmp_path: Path, relative: str
) -> None:
    root = _write_valid_project(tmp_path)
    (root / relative).unlink()

    result = _run(root)

    assert not result.ok
    assert any(relative in error for error in result.errors)


def test_release_check_rejects_dot_env(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    (root / ".env").write_text("PLACEHOLDER=value\n", encoding="utf-8")

    result = _run(root)

    assert not result.ok
    assert any(".env" in error for error in result.errors)


@pytest.mark.parametrize("name", ("id_rsa", "id_ed25519", "deploy.pem", "signing.key"))
def test_release_check_rejects_private_key_files(tmp_path: Path, name: str) -> None:
    root = _write_valid_project(tmp_path)
    (root / name).write_text("not a real key\n", encoding="utf-8")

    result = _run(root)

    assert not result.ok
    assert any(name in error and "secret-like" in error for error in result.errors)


def test_release_check_rejects_embedded_private_key_material(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    marker = "-----BEGIN " + "PRIVATE KEY-----"
    (root / "notes.txt").write_text(marker + "\nnot-a-real-key\n", encoding="utf-8")

    result = _run(root)

    assert not result.ok
    assert any("notes.txt" in error and "credential" in error for error in result.errors)


@pytest.mark.parametrize(
    "payload",
    (
        "hf_" + "a" * 24,
        "github_" + "pat_" + "a" * 24,
        "api_" + "key=" + "a" * 24,
    ),
)
def test_release_check_rejects_common_credential_patterns(
    tmp_path: Path, payload: str
) -> None:
    root = _write_valid_project(tmp_path)
    (root / "settings.txt").write_text(payload + "\n", encoding="utf-8")

    result = _run(root)

    assert not result.ok
    assert any("possible credential" in error for error in result.errors)


def test_release_check_rejects_raw_dataset_files(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    raw = root / "data" / "raw" / "sample.jpg"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"fixture")

    result = _run(root)

    assert not result.ok
    assert any("raw dataset" in error and "sample.jpg" in error for error in result.errors)


def test_release_check_rejects_files_above_100_mib(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    oversized = root / "oversized.bin"
    with oversized.open("wb") as stream:
        stream.seek(100 * 1024 * 1024)
        stream.write(b"\0")

    result = _run(root)

    assert not result.ok
    assert any("100 MiB" in error and "oversized.bin" in error for error in result.errors)


@pytest.mark.parametrize("missing_field", MODEL_CARD_FIELDS)
def test_release_check_rejects_each_missing_model_card_field(
    tmp_path: Path, missing_field: str
) -> None:
    root = _write_valid_project(tmp_path)
    model_card = "".join(
        f"## {field.title()}\nDocumented.\n"
        for field in MODEL_CARD_FIELDS
        if field != missing_field
    )
    (root / "docs" / "model-card.md").write_text(model_card, encoding="utf-8")

    result = _run(root)

    assert not result.ok
    assert any(f"model card missing field: {missing_field}" == error for error in result.errors)


@pytest.mark.parametrize("relative", ("artifacts/best.pt", "export/model.onnx"))
def test_release_check_rejects_checkpoint_and_onnx_binaries(
    tmp_path: Path, relative: str
) -> None:
    root = _write_valid_project(tmp_path)
    binary = root / relative
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary")

    result = _run(root)

    assert not result.ok
    assert any("model binary" in error and relative in error for error in result.errors)


def test_release_check_rejects_missing_licence_declarations(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    (root / "LICENSE").write_text("Unknown\n", encoding="utf-8")
    (root / "THIRD_PARTY_NOTICES.md").write_text("Incomplete\n", encoding="utf-8")

    result = _run(root)

    assert not result.ok
    assert any("project licence" in error for error in result.errors)
    for identifier in ("Apache-2.0", "CC-BY-4.0", "MIT", "REQUIRES-VERIFICATION"):
        assert any(identifier in error for error in result.errors)


def test_release_check_skips_binary_and_undecodable_text_safely(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    (root / "binary.dat").write_bytes(b"\x00\xff\xfe\xfd\x80")

    result = _run(root)

    assert result.ok, result.errors


def test_release_check_uses_only_git_tracked_files_inside_repository(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", *REQUIRED_DOCUMENTS], check=True)
    raw = root / "data" / "raw" / "local-only.jpg"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"local dataset")
    (root / ".env").write_text("LOCAL_ONLY=value\n", encoding="utf-8")

    result = _run(root)

    assert result.ok, result.errors


def test_release_check_rejects_tracked_raw_data_inside_repository(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", *REQUIRED_DOCUMENTS], check=True)
    raw = root / "data" / "raw" / "tracked.jpg"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"tracked dataset")
    subprocess.run(["git", "-C", str(root), "add", "-f", "data/raw/tracked.jpg"], check=True)

    result = _run(root)

    assert not result.ok
    assert any("raw dataset" in error and "tracked.jpg" in error for error in result.errors)
