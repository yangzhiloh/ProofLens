from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from prooflens.data.licences import SID_SET
from prooflens.data.schema import ManifestRecord
from prooflens.errors import DataIntegrityError


class SidSetAdapter:
    """Normalize SID-Set metadata rows to the primary binary label policy."""

    def __init__(self, version: str, root: Path | None = None) -> None:
        self.version = version
        self.root = root

    def scan(self) -> Iterator[ManifestRecord]:
        raise DataIntegrityError(
            "SID-Set scanning requires verified metadata rows with source labels; use scan_rows()."
        )

    def scan_rows(self, rows: Iterator[Mapping[str, Any]] | list[Mapping[str, Any]]) -> Iterator[ManifestRecord]:
        for row in rows:
            label = int(row["label"])
            if label == 2:
                continue
            if label not in (0, 1):
                raise ValueError(f"SID-Set label must be 0, 1, or 2; received {label}")
            image_id = str(row["img_id"])
            yield self._record(
                image_id,
                label,
                Path(row["image_path"]),
                width=int(row.get("width", 1)),
                height=int(row.get("height", 1)),
                file_format=str(row.get("file_format", "UNKNOWN")),
            )

    def _record(
        self, image_id: str, label: int, path: Path, width: int = 1, height: int = 1, file_format: str = "UNKNOWN"
    ) -> ManifestRecord:
        return ManifestRecord(
            sample_id=image_id,
            path=path,
            label=label,
            dataset_name=SID_SET.dataset_name,
            dataset_version=self.version,
            generator_family="authentic" if label == 0 else "generated",
            source_group_id=image_id,
            original_image_id=image_id,
            width=width,
            height=height,
            file_format=file_format,
            licence_identifier=SID_SET.licence_identifier,
        )
