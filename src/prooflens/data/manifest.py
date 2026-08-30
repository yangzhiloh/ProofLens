from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

from prooflens.data.adapters.base import DatasetAdapter
from prooflens.data.hashing import enrich_hashes
from prooflens.data.schema import ManifestRecord, records_to_frame
from prooflens.errors import ManifestBuildError


@dataclass(frozen=True)
class ManifestBuildResult:
    output_path: Path
    valid_count: int
    corrupt_count: int
    corrupt_paths: tuple[Path, ...]


def build_manifest(
    adapters: Sequence[DatasetAdapter], output_path: Path, max_corrupt_fraction: float = 0.01
) -> ManifestBuildResult:
    """Validate image decoding, write Parquet atomically, and reject excess corruption."""
    if not 0 <= max_corrupt_fraction <= 1:
        raise ValueError("max_corrupt_fraction must be between 0 and 1")

    valid_records: list[ManifestRecord] = []
    corrupt_paths: list[Path] = []
    for adapter in adapters:
        for record in adapter.scan():
            try:
                valid_records.append(_validated_record(record))
            except (OSError, UnidentifiedImageError, ValueError):
                corrupt_paths.append(record.path)

    total_count = len(valid_records) + len(corrupt_paths)
    corrupt_fraction = len(corrupt_paths) / total_count if total_count else 0.0
    if corrupt_fraction > max_corrupt_fraction:
        raise ManifestBuildError(f"corrupt fraction {corrupt_fraction:.6f} exceeds {max_corrupt_fraction:.6f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    try:
        enrich_hashes(records_to_frame(valid_records)).to_parquet(temporary_path, index=False)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return ManifestBuildResult(
        output_path=output_path,
        valid_count=len(valid_records),
        corrupt_count=len(corrupt_paths),
        corrupt_paths=tuple(corrupt_paths),
    )


def _validated_record(record: ManifestRecord) -> ManifestRecord:
    with Image.open(record.path) as image:
        file_format = image.format or "UNKNOWN"
        decoded = ImageOps.exif_transpose(image).convert("RGB")
        width, height = decoded.size
    return record.model_copy(update={"width": width, "height": height, "file_format": file_format})
