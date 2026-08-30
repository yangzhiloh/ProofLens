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
            "# Third-party notices\n"
            "## DINOv2\nLicence: Apache-2.0\n"
            "## SID-Set\nLicence: CC-BY-4.0\n"
            "## CIFAKE\nLicence: MIT\n"
            "## WildFake\nLicence: REQUIRES-VERIFICATION\n"
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


def _run_fixture(root: Path):
    from scripts.release_check import discover_fixture_files, run_release_check

    return run_release_check(root, file_discovery=discover_fixture_files)


def _run_git(root: Path):
    from scripts.release_check import run_release_check

    return run_release_check(root)


@pytest.mark.parametrize("relative", REQUIRED_DOCUMENTS)
def test_release_check_rejects_each_missing_required_document(
    tmp_path: Path, relative: str
) -> None:
    root = _write_valid_project(tmp_path)
    (root / relative).unlink()

    result = _run_fixture(root)

    assert not result.ok
    assert any(relative in error for error in result.errors)


def test_release_check_rejects_dot_env(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    (root / ".env").write_text("PLACEHOLDER=value\n", encoding="utf-8")

    result = _run_fixture(root)

    assert not result.ok
    assert any(".env" in error for error in result.errors)


@pytest.mark.parametrize("name", ("id_rsa", "id_ed25519", "deploy.pem", "signing.key"))
def test_release_check_rejects_private_key_files(tmp_path: Path, name: str) -> None:
    root = _write_valid_project(tmp_path)
    (root / name).write_text("not a real key\n", encoding="utf-8")

    result = _run_fixture(root)

    assert not result.ok
    assert any(name in error and "secret-like" in error for error in result.errors)


@pytest.mark.parametrize(
    "key_prefix", ("", "RSA ", "EC ", "DSA ", "OPENSSH ", "ENCRYPTED ")
)
def test_release_check_rejects_embedded_private_key_material(
    tmp_path: Path, key_prefix: str
) -> None:
    root = _write_valid_project(tmp_path)
    marker = "-----BEGIN " + key_prefix + "PRIVATE KEY-----"
    (root / "notes.txt").write_text(marker + "\nnot-a-real-key\n", encoding="utf-8")

    result = _run_fixture(root)

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

    result = _run_fixture(root)

    assert not result.ok
    assert any("possible credential" in error for error in result.errors)


def test_release_check_rejects_raw_dataset_files(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    raw = root / "data" / "raw" / "sample.jpg"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"fixture")

    result = _run_fixture(root)

    assert not result.ok
    assert any("raw dataset" in error and "sample.jpg" in error for error in result.errors)


def test_release_check_rejects_files_above_100_mib(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    oversized = root / "oversized.bin"
    with oversized.open("wb") as stream:
        stream.seek(100 * 1024 * 1024)
        stream.write(b"\0")

    result = _run_fixture(root)

    assert not result.ok
    assert any("100 MiB" in error and "oversized.bin" in error for error in result.errors)


def test_release_check_scans_text_files_larger_than_two_mib(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    credential = ("hf_" + "a" * 24).encode()
    (root / "large.txt").write_bytes(b"x" * (2 * 1024 * 1024 + 1) + credential)

    result = _run_fixture(root)

    assert not result.ok
    assert "possible credential in: large.txt" in result.errors


def test_release_check_detects_credential_split_across_scan_chunks(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    credential = ("hf_" + "b" * 24).encode()
    prefix = b"x" * (64 * 1024 - 2)
    (root / "boundary.txt").write_bytes(prefix + credential)

    result = _run_fixture(root)

    assert not result.ok
    assert "possible credential in: boundary.txt" in result.errors


def test_release_check_detects_credential_before_nul_byte(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    credential = ("hf_" + "c" * 24).encode()
    (root / "nul.dat").write_bytes(credential + b"\x00binary payload")

    result = _run_fixture(root)

    assert not result.ok
    assert "possible credential in: nul.dat" in result.errors


@pytest.mark.parametrize("encoding", ("utf-16-le", "utf-16-be"))
def test_release_check_detects_utf16_like_credential_bytes(
    tmp_path: Path, encoding: str
) -> None:
    root = _write_valid_project(tmp_path)
    credential = "hf_" + "d" * 24
    (root / "wide.dat").write_bytes(credential.encode(encoding))

    result = _run_fixture(root)

    assert not result.ok
    assert "possible credential in: wide.dat" in result.errors


def test_release_check_detects_credential_before_non_utf8_byte(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    credential = ("hf_" + "e" * 24).encode()
    (root / "undecodable.dat").write_bytes(credential + b"\xff")

    result = _run_fixture(root)

    assert not result.ok
    assert "possible credential in: undecodable.dat" in result.errors


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

    result = _run_fixture(root)

    assert not result.ok
    assert any(f"model card missing field: {missing_field}" == error for error in result.errors)


def test_release_check_rejects_model_card_words_outside_markdown_headings(
    tmp_path: Path,
) -> None:
    root = _write_valid_project(tmp_path)
    (root / "docs" / "model-card.md").write_text(
        "# Model card\nintended use limitations datasets metrics ethical\n",
        encoding="utf-8",
    )

    result = _run_fixture(root)

    assert not result.ok
    for field in MODEL_CARD_FIELDS:
        assert f"model card missing field: {field}" in result.errors


@pytest.mark.parametrize("relative", ("artifacts/best.pt", "export/model.onnx"))
def test_release_check_rejects_checkpoint_and_onnx_binaries(
    tmp_path: Path, relative: str
) -> None:
    root = _write_valid_project(tmp_path)
    binary = root / relative
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"binary")

    result = _run_fixture(root)

    assert not result.ok
    assert any("model binary" in error and relative in error for error in result.errors)


def test_release_check_rejects_missing_licence_declarations(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    (root / "LICENSE").write_text("Unknown\n", encoding="utf-8")
    (root / "THIRD_PARTY_NOTICES.md").write_text("Incomplete\n", encoding="utf-8")

    result = _run_fixture(root)

    assert not result.ok
    assert any("project licence" in error for error in result.errors)
    for identifier in ("Apache-2.0", "CC-BY-4.0", "MIT", "REQUIRES-VERIFICATION"):
        assert any(identifier in error for error in result.errors)


def test_release_check_rejects_unrelated_third_party_licence_identifiers(
    tmp_path: Path,
) -> None:
    root = _write_valid_project(tmp_path)
    (root / "THIRD_PARTY_NOTICES.md").write_text(
        "# Third-party notices\n"
        "## DINOv2\nComponent only.\n"
        "## SID-Set\nComponent only.\n"
        "## CIFAKE\nComponent only.\n"
        "## WildFake\nComponent only.\n"
        "## Unrelated identifiers\nApache-2.0 CC-BY-4.0 MIT REQUIRES-VERIFICATION\n",
        encoding="utf-8",
    )

    result = _run_fixture(root)

    assert not result.ok
    for component in ("DINOv2", "SID-Set", "CIFAKE", "WildFake"):
        assert any(component in error for error in result.errors)


def test_release_check_rejects_one_line_third_party_notices(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    (root / "THIRD_PARTY_NOTICES.md").write_text(
        "DINOv2 Apache-2.0 SID-Set CC-BY-4.0 CIFAKE MIT WildFake REQUIRES-VERIFICATION\n",
        encoding="utf-8",
    )

    result = _run_fixture(root)

    assert not result.ok
    assert len([error for error in result.errors if "licence association" in error]) == 4


def test_release_check_accepts_valid_markdown_release_documents(tmp_path: Path) -> None:
    result = _run_fixture(_write_valid_project(tmp_path))

    assert result.ok, result.errors


def test_release_check_skips_binary_and_undecodable_text_safely(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    (root / "binary.dat").write_bytes(b"\x00\xff\xfe\xfd\x80")

    result = _run_fixture(root)

    assert result.ok, result.errors


def test_release_check_uses_only_git_tracked_files_inside_repository(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", *REQUIRED_DOCUMENTS], check=True)
    raw = root / "data" / "raw" / "local-only.jpg"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"local dataset")
    (root / ".env").write_text("LOCAL_ONLY=value\n", encoding="utf-8")

    result = _run_git(root)

    assert result.ok, result.errors


def test_release_check_supports_explicit_non_git_fixture_discovery(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)

    result = _run_fixture(root)

    assert result.ok, result.errors


def test_release_check_fails_closed_outside_git_without_fixture_discovery(
    tmp_path: Path,
) -> None:
    root = _write_valid_project(tmp_path)

    result = _run_git(root)

    assert not result.ok
    assert any("Git worktree" in error for error in result.errors)


def test_release_check_rejects_nested_root_inside_git_worktree(tmp_path: Path) -> None:
    repository = _write_valid_project(tmp_path / "repository")
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    subprocess.run(["git", "-C", str(repository), "add", *REQUIRED_DOCUMENTS], check=True)
    nested = _write_valid_project(repository / "nested")

    result = _run_git(nested)

    assert not result.ok
    assert any("top level" in error for error in result.errors)


def test_release_check_rejects_untracked_required_document(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    tracked = [relative for relative in REQUIRED_DOCUMENTS if relative != "docs/model-card.md"]
    subprocess.run(["git", "-C", str(root), "add", *tracked], check=True)

    result = _run_git(root)

    assert not result.ok
    assert any("required document is not tracked: docs/model-card.md" == error for error in result.errors)


def test_release_check_fails_closed_when_git_listing_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.release_check as release_module

    root = _write_valid_project(tmp_path)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", *REQUIRED_DOCUMENTS], check=True)
    real_run = subprocess.run

    def fail_listing(args, **kwargs):
        if args[-2:] == ["ls-files", "-z"]:
            return subprocess.CompletedProcess(args, 1, stdout=b"", stderr=b"listing failed")
        return real_run(args, **kwargs)

    monkeypatch.setattr(release_module.subprocess, "run", fail_listing)

    result = release_module.run_release_check(root)

    assert not result.ok
    assert any("git ls-files failed" in error for error in result.errors)


def test_release_check_reports_git_paths_relative_to_repository_root(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    secret = root / "nested" / ".env"
    secret.parent.mkdir()
    secret.write_text("LOCAL=value\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(root), "add", *REQUIRED_DOCUMENTS, "nested/.env"], check=True
    )

    result = _run_git(root)

    assert not result.ok
    assert "secret-like file must not be published: nested/.env" in result.errors


def test_release_check_rejects_tracked_raw_data_inside_repository(tmp_path: Path) -> None:
    root = _write_valid_project(tmp_path)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", *REQUIRED_DOCUMENTS], check=True)
    raw = root / "data" / "raw" / "tracked.jpg"
    raw.parent.mkdir(parents=True)
    raw.write_bytes(b"tracked dataset")
    subprocess.run(["git", "-C", str(root), "add", "-f", "data/raw/tracked.jpg"], check=True)

    result = _run_git(root)

    assert not result.ok
    assert any("raw dataset" in error and "tracked.jpg" in error for error in result.errors)
