from __future__ import annotations

import hashlib
from os import PathLike
from pathlib import Path

import imagehash
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

from prooflens.errors import DataIntegrityError

PathValue = str | PathLike[str]


def sha256_file(path: PathValue, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of the exact file bytes."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise DataIntegrityError("hash chunk_size must be a positive integer")
    resolved = _coerce_path(path, "hash path")
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as stream:
            for chunk in iter(lambda: stream.read(chunk_size), b""):
                digest.update(chunk)
    except (OSError, ValueError) as error:
        raise DataIntegrityError(f"cannot hash file bytes at {resolved}: {error}") from error
    return digest.hexdigest()


def perceptual_hash_file(path: PathValue) -> str:
    """Return a 64-bit pHash after EXIF orientation and RGB normalization."""
    resolved = _coerce_path(path, "perceptual hash path")
    try:
        with Image.open(resolved) as image:
            normalized = ImageOps.exif_transpose(image).convert("RGB")
            return str(imagehash.phash(normalized))
    except (OSError, UnidentifiedImageError, ValueError) as error:
        raise DataIntegrityError(f"cannot perceptually hash image at {resolved}: {error}") from error


def enrich_hashes(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with exact and perceptual hashes for every local image path."""
    if "path" not in frame.columns:
        raise DataIntegrityError("hash enrichment requires the path column")
    result = frame.copy(deep=True)
    exact: list[str] = []
    perceptual: list[str] = []
    for index, value in result["path"].items():
        path = _coerce_path(value, f"path at row {index}")
        exact.append(sha256_file(path))
        perceptual.append(perceptual_hash_file(path))
    result["content_checksum"] = exact
    result["perceptual_hash"] = perceptual
    return result


def _coerce_path(value: object, context: str) -> Path:
    if isinstance(value, str):
        if not value.strip():
            raise DataIntegrityError(f"{context} must be a nonempty local path")
    elif not isinstance(value, PathLike):
        raise DataIntegrityError(f"{context} must be a nonempty local path")
    try:
        return Path(value)
    except (OSError, TypeError, ValueError) as error:
        raise DataIntegrityError(f"{context} must be a nonempty local path") from error
