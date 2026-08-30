from collections.abc import Iterator
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from prooflens.data.schema import ManifestRecord
from prooflens.errors import DataIntegrityError


class CanonicalParquetAdapter:
    """Read an acquired canonical Parquet manifest without changing its records."""

    def __init__(self, manifest_path: Path, expected_dataset_name: str) -> None:
        self.manifest_path = manifest_path
        self.expected_dataset_name = expected_dataset_name

    def scan(self) -> Iterator[ManifestRecord]:
        if not self.manifest_path.is_file():
            raise DataIntegrityError(f"canonical manifest is missing: {self.manifest_path}")
        for row in pd.read_parquet(self.manifest_path).to_dict(orient="records"):
            try:
                record = ManifestRecord.model_validate(row)
            except ValidationError as error:
                raise DataIntegrityError("canonical manifest row has an invalid schema") from error
            if record.dataset_name != self.expected_dataset_name:
                raise DataIntegrityError(
                    f"canonical manifest record has dataset_name {record.dataset_name!r}; "
                    f"expected {self.expected_dataset_name!r}"
                )
            yield record
