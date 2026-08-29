"""Validate that the public release contains documentation and no tracked secrets or data."""

from __future__ import annotations

import argparse
import codecs
import os
import re
import subprocess
from collections.abc import Callable
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
THIRD_PARTY_ASSOCIATIONS = (
    ("DINOv2", "Apache-2.0"),
    ("SID-Set", "CC-BY-4.0"),
    ("CIFAKE", "MIT"),
    ("WildFake", "REQUIRES-VERIFICATION"),
)
MAX_RELEASE_BYTES = 100 * 1024 * 1024
TEXT_SCAN_CHUNK_BYTES = 64 * 1024
TEXT_SCAN_OVERLAP = 512

_FORBIDDEN_NAMES = frozenset({".env", "id_rsa", "id_ed25519"})
_PRIVATE_KEY_SUFFIXES = frozenset({".pem", ".key"})
_MODEL_BINARY_SUFFIXES = frozenset({".ckpt", ".onnx", ".pt", ".pth", ".safetensors"})
_PRIVATE_KEY_PATTERN = re.compile(
    r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"
)
_TOKEN_PATTERN = re.compile(
    r"(?i)(hf_[a-z0-9]{20,}|github_pat_[a-z0-9_]{20,}|"
    r"api[_-]?key\s*[=:]\s*[^\s\"'<>]{8,})"
)
_MARKDOWN_HEADING_PATTERN = re.compile(
    r"(?m)^#{1,6}[ \t]+(.+?)[ \t]*#*[ \t]*$"
)


@dataclass(frozen=True, slots=True)
class ReleaseCheckResult:
    ok: bool
    errors: tuple[str, ...]


FileDiscovery = Callable[[Path], tuple[tuple[str, Path], ...]]


class ReleaseDiscoveryError(RuntimeError):
    """Raised when the publishable Git file set cannot be proven."""


def run_release_check(
    root: Path, *, file_discovery: FileDiscovery | None = None
) -> ReleaseCheckResult:
    """Check required release files and inspect publishable files beneath ``root``."""

    root = Path(root).resolve()
    errors: list[str] = []
    for relative in REQUIRED_DOCUMENTS:
        if not (root / relative).is_file():
            errors.append(f"missing required document: {relative}")

    discover = _release_files if file_discovery is None else file_discovery
    discovery_failed = False
    try:
        release_files = discover(root)
    except ReleaseDiscoveryError as error:
        errors.append(str(error))
        release_files = ()
        discovery_failed = True
    if file_discovery is None and not discovery_failed:
        tracked = {relative for relative, _path in release_files}
        for relative in REQUIRED_DOCUMENTS:
            if relative not in tracked:
                errors.append(f"required document is not tracked: {relative}")
    for relative, path in release_files:
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
        if size <= MAX_RELEASE_BYTES and _contains_credential(path):
            errors.append(f"possible credential in: {relative}")

    _check_licence_declarations(root, errors)
    _check_model_card(root, errors)
    unique_errors = tuple(sorted(set(errors)))
    return ReleaseCheckResult(ok=not unique_errors, errors=unique_errors)


def _release_files(root: Path) -> tuple[tuple[str, Path], ...]:
    tracked = _git_ls_files(root)
    if tracked is None:
        raise ReleaseDiscoveryError("release root must be a Git worktree top level")
    return tuple((relative, root / Path(relative)) for relative in tracked)


def discover_fixture_files(root: Path) -> tuple[tuple[str, Path], ...]:
    """Recursively discover isolated non-Git test-fixture files."""

    root = Path(root).resolve()
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
        top_level = subprocess.run(
            ["git", "-C", os.fspath(root), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    if top_level.returncode != 0:
        return None
    discovered_root = Path(top_level.stdout.strip()).resolve()
    if discovered_root != root:
        raise ReleaseDiscoveryError(
            f"release root must be the Git worktree top level: {discovered_root}"
        )

    try:
        listed = subprocess.run(
            ["git", "-C", os.fspath(root), "ls-files", "-z"],
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise ReleaseDiscoveryError(f"git ls-files failed: {error}") from error
    if listed.returncode != 0:
        detail = os.fsdecode(listed.stderr).strip() or f"exit code {listed.returncode}"
        raise ReleaseDiscoveryError(f"git ls-files failed: {detail}")
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


def _contains_credential(path: Path) -> bool:
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    tail = ""
    found = False
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(TEXT_SCAN_CHUNK_BYTES):
                if b"\0" in chunk:
                    return False
                text = tail + decoder.decode(chunk)
                found = found or bool(
                    _TOKEN_PATTERN.search(text) or _PRIVATE_KEY_PATTERN.search(text)
                )
                tail = text[-TEXT_SCAN_OVERLAP:]
            final = tail + decoder.decode(b"", final=True)
    except (OSError, UnicodeDecodeError):
        return False
    return found or bool(_TOKEN_PATTERN.search(final) or _PRIVATE_KEY_PATTERN.search(final))


def _check_licence_declarations(root: Path, errors: list[str]) -> None:
    licence = root / "LICENSE"
    if licence.is_file():
        text = _read_text_safely(licence)
        if text is None or "MIT License" not in text:
            errors.append("project licence declaration is not MIT")

    notices = root / "THIRD_PARTY_NOTICES.md"
    if notices.is_file():
        text = _read_text_safely(notices)
        sections = {} if text is None else _markdown_sections(text)
        for component, identifier in THIRD_PARTY_ASSOCIATIONS:
            body = sections.get(component.casefold(), "")
            if identifier not in body:
                errors.append(
                    "third-party notices missing licence association: "
                    f"{component} -> {identifier}"
                )


def _check_model_card(root: Path, errors: list[str]) -> None:
    path = root / "docs" / "model-card.md"
    if not path.is_file():
        return
    text = _read_text_safely(path)
    if text is None:
        errors.append("model card is not readable UTF-8 text")
        return
    headings = tuple(
        match.group(1).strip().casefold()
        for match in _MARKDOWN_HEADING_PATTERN.finditer(text)
    )
    for field in MODEL_CARD_FIELDS:
        if not any(heading == field or heading.startswith(field + " ") for heading in headings):
            errors.append(f"model card missing field: {field}")


def _markdown_sections(text: str) -> dict[str, str]:
    matches = list(_MARKDOWN_HEADING_PATTERN.finditer(text))
    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        heading = match.group(1).strip().casefold()
        sections[heading] = sections.get(heading, "") + text[match.end() : end]
    return sections


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
