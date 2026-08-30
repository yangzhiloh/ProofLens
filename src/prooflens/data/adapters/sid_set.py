from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from prooflens.data.licences import SID_SET
from prooflens.data.schema import MANIFEST_COLUMNS, ManifestRecord
from prooflens.errors import DataIntegrityError


class SidSetAdapter:
    """Normalize SID-Set metadata rows to the primary binary label policy."""

    def __init__(self, version: str, root: Path | None = None) -> None:
        self.version = version
        self.root = root

    def scan(self) -> Iterator[ManifestRecord]:
        if self.root is None:
            raise DataIntegrityError(
                "SID-Set scanning requires an acquired root containing manifest.parquet"
            )
        root = Path(self.root).resolve()
        manifest_path = root / "manifest.parquet"
        if not manifest_path.is_file():
            raise DataIntegrityError(
                f"SID-Set acquired manifest is missing: {manifest_path}. "
                "Run the acquire command before building the primary manifest."
            )
        try:
            frame = pd.read_parquet(manifest_path)
        except Exception as error:
            raise DataIntegrityError(
                f"SID-Set acquired manifest is unreadable: {manifest_path}"
            ) from error
        missing = set(MANIFEST_COLUMNS) - set(frame.columns)
        if missing:
            raise DataIntegrityError(
                f"SID-Set acquired manifest is missing columns: {sorted(missing)}"
            )
        for position, raw in frame.iterrows():
            try:
                record = ManifestRecord.model_validate(raw.to_dict())
            except ValidationError as error:
                raise DataIntegrityError(
                    f"SID-Set acquired manifest row {position} is invalid"
                ) from error
            if record.dataset_name != SID_SET.dataset_name:
                raise DataIntegrityError(
                    f"SID-Set acquired manifest row {position} has dataset_name "
                    f"{record.dataset_name!r}"
                )
            candidate = record.path if record.path.is_absolute() else root / record.path
            resolved_path = candidate.resolve()
            try:
                resolved_path.relative_to(root)
            except ValueError as error:
                raise DataIntegrityError(
                    f"SID-Set acquired manifest row {position} points outside {root}"
                ) from error
            yield record.model_copy(update={"path": resolved_path})

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
