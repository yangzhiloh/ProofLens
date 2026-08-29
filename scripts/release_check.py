"""Validate that the public release contains documentation and no tracked secrets or data."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

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
THIRD_PARTY_IDENTIFIERS = ("Apache-2.0", "CC-BY-4.0", "MIT", "REQUIRES-VERIFICATION")
MAX_RELEASE_BYTES = 100 * 1024 * 1024
MAX_TEXT_SCAN_BYTES = 2 * 1024 * 1024

_FORBIDDEN_NAMES = frozenset({".env", "id_rsa", "id_ed25519"})
_PRIVATE_KEY_SUFFIXES = frozenset({".pem", ".key"})
_MODEL_BINARY_SUFFIXES = frozenset({".ckpt", ".onnx", ".pt", ".pth", ".safetensors"})
_PRIVATE_KEY_MARKER = "-----BEGIN " + "PRIVATE KEY-----"
_TOKEN_PATTERN = re.compile(
    r"(?i)(hf_[a-z0-9]{20,}|github_pat_[a-z0-9_]{20,}|"
    r"api[_-]?key\s*[=:]\s*[^\s\"'<>]{8,})"
)


@dataclass(frozen=True, slots=True)
class ReleaseCheckResult:
    ok: bool
    errors: tuple[str, ...]


def run_release_check(root: Path) -> ReleaseCheckResult:
    """Check required release files and inspect publishable files beneath ``root``."""

    root = Path(root).resolve()
    errors: list[str] = []
    for relative in REQUIRED_DOCUMENTS:
        if not (root / relative).is_file():
            errors.append(f"missing required document: {relative}")

    for relative, path in _release_files(root):
        try:
            size = path.stat().st_size
        except OSError as error:
            errors.append(f"unable to inspect release file {relative}: {error}")
            continue

        if path.name.casefold() in _FORBIDDEN_NAMES or path.suffix.casefold() in _PRIVATE_KEY_SUFFIXES:
            errors.append(f"secret-like file must not be published: {relative}")
        if size > MAX_RELEASE_BYTES:
            errors.append(f"file exceeds 100 MiB release limit: {relative}")
        if relative.casefold().startswith("data/raw/"):
            errors.append(f"raw dataset file must not be published: {relative}")
        if path.suffix.casefold() in _MODEL_BINARY_SUFFIXES:
            errors.append(f"model binary must be distributed outside git: {relative}")
        if size <= MAX_TEXT_SCAN_BYTES:
            text = _read_text_safely(path)
            if text is not None and (
                _TOKEN_PATTERN.search(text) or _PRIVATE_KEY_MARKER in text
            ):
                errors.append(f"possible credential in: {relative}")

    _check_licence_declarations(root, errors)
    _check_model_card(root, errors)
    unique_errors = tuple(sorted(set(errors)))
    return ReleaseCheckResult(ok=not unique_errors, errors=unique_errors)


def _release_files(root: Path) -> tuple[tuple[str, Path], ...]:
    tracked = _git_ls_files(root)
    if tracked is not None:
        return tuple((relative, root / Path(relative)) for relative in tracked)

    ignored_parts = {".git", ".venv", "__pycache__"}
    files: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        try:
            is_file = path.is_file()
        except OSError:
            is_file = False
        if not is_file:
            continue
        relative_path = path.relative_to(root)
        if ignored_parts.intersection(relative_path.parts):
            continue
        files.append((relative_path.as_posix(), path))
    return tuple(sorted(files))


def _git_ls_files(root: Path) -> tuple[str, ...] | None:
    try:
        inside = subprocess.run(
            ["git", "-C", os.fspath(root), "rev-parse", "--is-inside-work-tree"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        return None

    listed = subprocess.run(
        ["git", "-C", os.fspath(root), "ls-files", "-z"],
        check=False,
        capture_output=True,
    )
    if listed.returncode != 0:
        return ()
    return tuple(
        os.fsdecode(value)
        for value in listed.stdout.split(b"\0")
        if value
    )


def _read_text_safely(path: Path) -> str | None:
    try:
        data = path.read_bytes()
        if b"\0" in data:
            return None
        return data.decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _check_licence_declarations(root: Path, errors: list[str]) -> None:
    licence = root / "LICENSE"
    if licence.is_file():
        text = _read_text_safely(licence)
        if text is None or "MIT License" not in text:
            errors.append("project licence declaration is not MIT")

    notices = root / "THIRD_PARTY_NOTICES.md"
    if notices.is_file():
        text = _read_text_safely(notices)
        for identifier in THIRD_PARTY_IDENTIFIERS:
            if text is None or identifier not in text:
                errors.append(f"third-party notices missing licence identifier: {identifier}")


def _check_model_card(root: Path, errors: list[str]) -> None:
    path = root / "docs" / "model-card.md"
    if not path.is_file():
        return
    text = _read_text_safely(path)
    if text is None:
        errors.append("model card is not readable UTF-8 text")
        return
    lowered = text.casefold()
    for field in MODEL_CARD_FIELDS:
        if field not in lowered:
            errors.append(f"model card missing field: {field}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check tracked ProofLens release files")
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = run_release_check(args.root)
    if result.ok:
        print("release check: OK")
        raise SystemExit(0)
    print("release check: FAILED")
    for error in result.errors:
        print(f"- {error}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
