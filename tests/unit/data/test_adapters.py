from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from prooflens.data.adapters.cifake import CifakeAdapter
from prooflens.data.adapters.sid_set import SidSetAdapter
from prooflens.data.adapters.wildfake import WildFakeAdapter
from prooflens.data.manifest import build_manifest
from prooflens.data.schema import MANIFEST_COLUMNS, ManifestRecord, records_to_frame
from prooflens.errors import DataIntegrityError, ManifestBuildError


class StaticAdapter:
    def __init__(self, records: list[ManifestRecord]) -> None:
        self.records = records

    def scan(self) -> Iterator[ManifestRecord]:
        yield from self.records


def _record(path: Path, sample_id: str) -> ManifestRecord:
    return ManifestRecord(
        sample_id=sample_id, path=path, label=0, dataset_name="fixture", dataset_version="v1",
        generator_family="authentic", source_group_id=sample_id, original_image_id=sample_id,
        width=1, height=1, file_format="JPEG", licence_identifier="CC0-1.0",
    )


@pytest.fixture
def wildfake_fixture(tmp_path: Path) -> Path:
    real = tmp_path / "real" / "camera"
    fake = tmp_path / "fake" / "sdxl"
    real.mkdir(parents=True)
    fake.mkdir(parents=True)
    Image.new("RGB", (4, 3), "white").save(real / "real.jpg")
    Image.new("RGB", (4, 3), "black").save(fake / "fake.jpg")
    return tmp_path


@pytest.fixture
def valid_adapter(tmp_path: Path) -> StaticAdapter:
    path = tmp_path / "valid.jpg"
    Image.new("RGB", (4, 3), "white").save(path)
    return StaticAdapter([_record(path, "valid")])


@pytest.fixture
def corrupt_adapter(tmp_path: Path) -> StaticAdapter:
    path = tmp_path / "corrupt.jpg"
    path.write_bytes(b"not an image")
    return StaticAdapter([_record(path, "corrupt")])


def test_sid_adapter_excludes_tampered_label_two(tmp_path: Path) -> None:
    rows = [
        {"img_id": "real-1", "label": 0, "image_path": tmp_path / "real.jpg"},
        {"img_id": "full_synthetic_1", "label": 1, "image_path": tmp_path / "fake.jpg"},
        {"img_id": "tampered_1", "label": 2, "image_path": tmp_path / "edit.jpg"},
    ]
    records = list(SidSetAdapter(version="main").scan_rows(rows))
    assert [record.label for record in records] == [0, 1]


def test_sid_adapter_rejects_root_scanning_without_acquired_manifest(tmp_path: Path) -> None:
    (tmp_path / "tampered").mkdir()
    with pytest.raises(DataIntegrityError, match="acquired manifest is missing"):
        list(SidSetAdapter(version="main", root=tmp_path).scan())


def test_sid_adapter_rejects_acquired_rows_from_another_dataset(tmp_path: Path) -> None:
    path = tmp_path / "inside.jpg"
    Image.new("RGB", (4, 3), "white").save(path)
    records_to_frame([_record(path, "wrong-dataset")]).to_parquet(
        tmp_path / "manifest.parquet",
        index=False,
    )

    with pytest.raises(DataIntegrityError, match="dataset_name 'fixture'"):
        list(SidSetAdapter(version="main", root=tmp_path).scan())


def test_sid_adapter_rejects_acquired_paths_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "sid"
    root.mkdir()
    outside = tmp_path / "outside.jpg"
    Image.new("RGB", (4, 3), "white").save(outside)
    record = _record(outside, "outside").model_copy(
        update={"dataset_name": "sid_set"}
    )
    records_to_frame([record]).to_parquet(root / "manifest.parquet", index=False)

    with pytest.raises(DataIntegrityError, match="points outside"):
        list(SidSetAdapter(version="main", root=root).scan())


def test_canonical_parquet_adapter_yields_validated_records_without_changing_paths_or_labels(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.parquet"
    expected = [
        ManifestRecord(
            sample_id="sid-real",
            path=tmp_path / "images" / "real.png",
            label=0,
            dataset_name="sid_set",
            dataset_version="pinned-revision",
            generator_family="authentic",
            source_group_id="sid-real",
            original_image_id="sid-real",
            width=4,
            height=3,
            file_format="PNG",
            licence_identifier="CC-BY-4.0",
        ),
        ManifestRecord(
            sample_id="sid-fake",
            path=tmp_path / "images" / "fake.png",
            label=1,
            dataset_name="sid_set",
            dataset_version="pinned-revision",
            generator_family="generated",
            source_group_id="sid-fake",
            original_image_id="sid-fake",
            width=4,
            height=3,
            file_format="PNG",
            licence_identifier="CC-BY-4.0",
        ),
    ]
    pd.DataFrame([record.model_dump(mode="json") for record in expected]).to_parquet(
        manifest_path, index=False
    )

    from prooflens.data.adapters.local_manifest import CanonicalParquetAdapter

    actual = list(CanonicalParquetAdapter(manifest_path, "sid_set").scan())

    assert actual == expected


def test_wildfake_adapter_reads_generator_from_hierarchy(wildfake_fixture: Path) -> None:
    records = list(WildFakeAdapter(wildfake_fixture).scan())
    assert {record.generator_family for record in records if record.label == 1} == {"sdxl"}


def test_wildfake_adapter_missing_root_includes_manual_acquisition_guidance(
    tmp_path: Path,
) -> None:
    with pytest.raises(DataIntegrityError) as caught:
        list(WildFakeAdapter(tmp_path / "missing").scan())

    message = str(caught.value)
    assert "manual" in message.lower()
    assert "https://github.com/hy-zpg/AIGC-Image-Detection-Dataset" in message
    assert "https://modelscope.cn/datasets/hy2628982280/WildFake/summary" in message


def test_wildfake_adapter_rejects_missing_configured_directory(tmp_path: Path) -> None:
    (tmp_path / "real").mkdir()
    with pytest.raises(DataIntegrityError, match="fake"):
        list(WildFakeAdapter(tmp_path).scan())


def test_wildfake_adapter_rejects_empty_real_class(wildfake_fixture: Path) -> None:
    for path in (wildfake_fixture / "real").rglob("*"):
        if path.is_file():
            path.unlink()
    with pytest.raises(DataIntegrityError, match="real"):
        list(WildFakeAdapter(wildfake_fixture).scan())


def test_cifake_adapter_marks_records_as_stress_only(tmp_path: Path) -> None:
    for name, color in (("REAL", "white"), ("FAKE", "black")):
        directory = tmp_path / name
        directory.mkdir()
        Image.new("RGB", (4, 3), color).save(directory / f"{name.lower()}.png")
    records = list(CifakeAdapter(tmp_path).scan())
    assert {(record.label, record.dataset_name) for record in records} == {
        (0, "cifake_stress"), (1, "cifake_stress")
    }
    assert next(record for record in records if record.label == 1).generator_family == "stable-diffusion-1.4"


def test_cifake_adapter_rejects_missing_required_label_directory(tmp_path: Path) -> None:
    (tmp_path / "REAL").mkdir()
    with pytest.raises(DataIntegrityError, match="FAKE"):
        list(CifakeAdapter(tmp_path).scan())


def test_cifake_adapter_rejects_empty_required_class(tmp_path: Path) -> None:
    (tmp_path / "REAL").mkdir()
    fake = tmp_path / "FAKE"
    fake.mkdir()
    Image.new("RGB", (4, 3), "black").save(fake / "fake.png")
    with pytest.raises(DataIntegrityError, match="REAL"):
        list(CifakeAdapter(tmp_path).scan())


def test_manifest_builder_stops_above_corrupt_limit(
    valid_adapter: StaticAdapter, corrupt_adapter: StaticAdapter, tmp_path: Path
) -> None:
    with pytest.raises(ManifestBuildError, match="corrupt fraction"):
        build_manifest(
            [valid_adapter, corrupt_adapter], tmp_path / "manifest.parquet", max_corrupt_fraction=0.01
        )


def test_manifest_builder_persists_decoded_records_atomically(
    valid_adapter: StaticAdapter, tmp_path: Path
) -> None:
    output_path = tmp_path / "manifest.parquet"
    output_path.write_bytes(b"previous manifest")

    result = build_manifest([valid_adapter], output_path)

    frame = pd.read_parquet(output_path)
    assert result.valid_count == 1
    assert tuple(frame.columns) == MANIFEST_COLUMNS
    assert frame.loc[0, "width"] == 4
    assert frame.loc[0, "height"] == 3
    assert frame.loc[0, "file_format"] == "JPEG"
    assert len(frame.loc[0, "content_checksum"]) == 64
    assert len(frame.loc[0, "perceptual_hash"]) == 16
    assert not list(tmp_path.glob(".manifest.parquet.*.tmp"))


def test_manifest_builder_preserves_destination_when_parquet_write_fails(
    valid_adapter: StaticAdapter, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "manifest.parquet"
    original_bytes = b"previous manifest"
    output_path.write_bytes(original_bytes)

    def fail_write(self: pd.DataFrame, *args: object, **kwargs: object) -> None:
        raise OSError("disk failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_write)

    with pytest.raises(OSError, match="disk failure"):
        build_manifest([valid_adapter], output_path)

    assert output_path.read_bytes() == original_bytes
    assert not list(tmp_path.glob(".manifest.parquet.*.tmp"))
