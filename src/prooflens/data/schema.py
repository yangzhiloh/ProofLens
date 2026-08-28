from pathlib import Path
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class ManifestRecord(BaseModel):
    """One validated candidate for the canonical image manifest."""

    model_config = ConfigDict(extra="forbid")

    sample_id: str
    path: Path
    label: Literal[0, 1]
    dataset_name: str
    dataset_version: str
    generator_family: str
    source_group_id: str
    original_image_id: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    file_format: str
    licence_identifier: str
    content_checksum: str = ""
    perceptual_hash: str = ""
    split: str = "unassigned"


MANIFEST_COLUMNS = tuple(ManifestRecord.model_fields)


def records_to_frame(records: list[ManifestRecord]) -> pd.DataFrame:
    """Serialize records with the canonical, declaration-order column layout."""

    return pd.DataFrame(
        [record.model_dump(mode="json") for record in records], columns=MANIFEST_COLUMNS
    )
