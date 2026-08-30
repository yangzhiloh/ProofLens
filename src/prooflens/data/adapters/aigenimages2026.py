from __future__ import annotations

import csv
import unicodedata
from collections.abc import Iterator
from pathlib import Path

from PIL import Image

from prooflens.data.licences import AIGENIMAGES2026
from prooflens.data.schema import ManifestRecord
from prooflens.errors import DataIntegrityError


class AIGenImages2026Adapter:
    """Adapt the small paired evaluation release with generator-aware grouping."""

    def __init__(self, root: Path, version: str) -> None:
        self.root = Path(root)
        self.version = version

    def scan(self) -> Iterator[ManifestRecord]:
        pairs_path = self.root / "eval_real_pairs.csv"
        real_root = self.root / "0_real"
        fake_root = self.root / "1_fake"
        if not pairs_path.is_file() or not real_root.is_dir() or not fake_root.is_dir():
            raise DataIntegrityError(
                "AIGenImages2026 requires eval_real_pairs.csv, 0_real/, and 1_fake/"
            )
        fake_by_name, fake_by_normalized_name = _unique_file_indexes(fake_root)
        seen_real: set[str] = set()
        seen_fake: set[str] = set()
        with pairs_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {"real", "fake"} <= set(reader.fieldnames):
                raise DataIntegrityError(
                    "AIGenImages2026 pair CSV requires real and fake columns"
                )
            for position, row in enumerate(reader, start=2):
                real_name = str(row.get("real", "")).strip()
                fake_name = str(row.get("fake", "")).strip()
                if not real_name or not fake_name:
                    raise DataIntegrityError(
                        f"AIGenImages2026 pair row {position} has a blank filename"
                    )
                if real_name in seen_real or fake_name in seen_fake:
                    raise DataIntegrityError(
                        f"AIGenImages2026 pair row {position} reuses an image"
                    )
                real_path = real_root / real_name
                fake_path = fake_by_name.get(fake_name)
                if fake_path is None:
                    fake_path = fake_by_normalized_name.get(
                        _normalized_filename(fake_name)
                    )
                if not real_path.is_file() or fake_path is None:
                    raise DataIntegrityError(
                        f"AIGenImages2026 pair row {position} references a missing image"
                    )
                seen_real.add(real_name)
                seen_fake.add(fake_name)
                group_id = f"aigenimages2026-pair:{real_path.stem}"
                yield self._record(real_path, 0, "authentic", group_id)
                yield self._record(fake_path, 1, fake_path.parent.name, group_id)

    def _record(
        self, path: Path, label: int, generator_family: str, group_id: str
    ) -> ManifestRecord:
        try:
            with Image.open(path) as image:
                width, height = image.size
                file_format = str(image.format or path.suffix.lstrip(".")).upper()
        except (OSError, ValueError) as error:
            raise DataIntegrityError(
                f"AIGenImages2026 image is unreadable: {path}"
            ) from error
        relative = path.relative_to(self.root).as_posix()
        return ManifestRecord(
            sample_id=f"aigenimages2026:{relative}",
            path=path.resolve(),
            label=label,
            dataset_name=AIGENIMAGES2026.dataset_name,
            dataset_version=self.version,
            generator_family=generator_family,
            source_group_id=group_id,
            original_image_id=group_id,
            width=width,
            height=height,
            file_format=file_format,
            licence_identifier=AIGENIMAGES2026.licence_identifier,
        )


def _unique_file_indexes(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    result: dict[str, Path] = {}
    normalized: dict[str, Path] = {}
    for path in sorted(root.rglob("*"), key=lambda candidate: candidate.as_posix()):
        if not path.is_file():
            continue
        if path.name in result:
            raise DataIntegrityError(
                f"AIGenImages2026 has duplicate fake filename: {path.name}"
            )
        result[path.name] = path
        key = _normalized_filename(path.name)
        if key in normalized:
            raise DataIntegrityError(
                f"AIGenImages2026 has ambiguous normalized fake filename: {path.name}"
            )
        normalized[key] = path
    if not result:
        raise DataIntegrityError("AIGenImages2026 has no generated images")
    return result, normalized


def _normalized_filename(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value).casefold()
        if character.isalnum()
    )
