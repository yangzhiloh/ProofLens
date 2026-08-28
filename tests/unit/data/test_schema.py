from pathlib import Path

import pytest

from prooflens.data.schema import MANIFEST_COLUMNS, ManifestRecord, records_to_frame


@pytest.fixture
def valid_record() -> ManifestRecord:
    return ManifestRecord(
        sample_id="real-1", path=Path("image.jpg"), label=0, dataset_name="sid_set",
        dataset_version="main", generator_family="authentic", source_group_id="real-1",
        original_image_id="real-1", width=1024, height=1024, file_format="JPEG",
        licence_identifier="CC-BY-4.0",
    )


def test_manifest_rejects_non_binary_primary_label() -> None:
    with pytest.raises(ValueError):
        ManifestRecord(
            sample_id="tampered-1", path=Path("image.jpg"), label=2, dataset_name="sid_set",
            dataset_version="main", generator_family="tampered", source_group_id="tampered-1",
            original_image_id="tampered-1", width=1024, height=1024, file_format="JPEG",
            licence_identifier="CC-BY-4.0",
        )


def test_records_to_frame_has_stable_column_order(valid_record: ManifestRecord) -> None:
    frame = records_to_frame([valid_record])
    assert tuple(frame.columns) == MANIFEST_COLUMNS


def test_records_to_frame_preserves_schema_when_empty() -> None:
    assert tuple(records_to_frame([]).columns) == MANIFEST_COLUMNS
