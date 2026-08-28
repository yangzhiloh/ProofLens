from collections.abc import Iterator
from pathlib import Path

from prooflens.data.licences import CIFAKE
from prooflens.data.schema import ManifestRecord
from prooflens.errors import DataIntegrityError


class CifakeAdapter:
    """Adapt CIFAKE folders as a low-resolution stress-only dataset."""

    def __init__(self, root: Path, version: str = "main") -> None:
        self.root = root
        self.version = version

    def scan(self) -> Iterator[ManifestRecord]:
        if not self.root.is_dir():
            raise DataIntegrityError(f"CIFAKE root is missing: {self.root}. Mount the approved dataset root.")
        records: list[ManifestRecord] = []
        for directory_name, label, family in (
            ("REAL", 0, "authentic"),
            ("FAKE", 1, "stable-diffusion-1.4"),
        ):
            directory = self.root / directory_name
            if not directory.is_dir():
                raise DataIntegrityError(
                    f"CIFAKE required directory is missing: {directory}. Expected REAL and FAKE folders."
                )
            paths = sorted((path for path in directory.rglob("*") if path.is_file()), key=lambda p: p.as_posix())
            if not paths:
                raise DataIntegrityError(f"CIFAKE {directory_name} directory has no images: {directory}")
            for path in paths:
                sample_id = path.relative_to(self.root).as_posix()
                records.append(ManifestRecord(
                    sample_id=sample_id,
                    path=path,
                    label=label,
                    dataset_name=CIFAKE.dataset_name,
                    dataset_version=self.version,
                    generator_family=family,
                    source_group_id=sample_id,
                    original_image_id=path.stem,
                    width=1,
                    height=1,
                    file_format="UNKNOWN",
                    licence_identifier=CIFAKE.licence_identifier,
                ))
        yield from records
