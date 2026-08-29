from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from prooflens.errors import DataIntegrityError, ImageDecodeError


def _dataset_type():
    from prooflens.data.dataset import SourceImageDataset

    return SourceImageDataset


def _manifest_frame(tmp_path: Path) -> pd.DataFrame:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.new("L", (6, 4), 80).save(first)
    Image.new("RGBA", (5, 3), (20, 40, 60, 120)).save(second)
    frame = pd.DataFrame(
        [
            {
                "sample_id": "sample-real",
                "path": first,
                "label": 0,
                "dataset_name": "sid_set",
                "dataset_version": "main",
                "generator_family": "authentic",
                "source_group_id": "source-real",
                "original_image_id": "original-real",
                "width": 1,
                "height": 1,
                "file_format": "PNG",
                "licence_identifier": "CC-BY-4.0",
                "content_checksum": "a" * 64,
                "perceptual_hash": "1" * 16,
                "split": "train",
                "split_group_id": "split-real",
            },
            {
                "sample_id": "sample-fake",
                "path": second,
                "label": 1,
                "dataset_name": "wildfake",
                "dataset_version": "main",
                "generator_family": "sdxl",
                "source_group_id": "source-fake",
                "original_image_id": "original-fake",
                "width": 999,
                "height": 999,
                "file_format": "PNG",
                "licence_identifier": "REQUIRES-VERIFICATION",
                "content_checksum": "b" * 64,
                "perceptual_hash": "2" * 16,
                "split": "validation",
                "split_group_id": "split-fake",
            },
        ],
        index=[73, 73],
    )
    return frame


def test_source_dataset_preserves_positional_order_and_returns_immutable_metadata(
    tmp_path: Path,
) -> None:
    frame = _manifest_frame(tmp_path)
    original = frame.copy(deep=True)

    dataset = _dataset_type()(frame)
    first = dataset[0]
    second = dataset[1]

    assert len(dataset) == 2
    assert (first.sample_id, second.sample_id) == ("sample-real", "sample-fake")
    assert first.image.mode == second.image.mode == "RGB"
    assert first.image.size == (6, 4)
    assert (
        second.label,
        second.dataset_name,
        second.generator_family,
        second.source_group_id,
        second.split,
        second.split_group_id,
    ) == (1, "wildfake", "sdxl", "source-fake", "validation", "split-fake")
    with pytest.raises(FrozenInstanceError):
        second.label = 0  # type: ignore[misc]
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (("drop", "path"), "required fields.*path"),
        (("set", "label", 2), "label.*binary"),
        (("set", "sample_id", "  "), "sample_id.*nonempty"),
        (("set", "split_group_id", None), "split_group_id.*nonempty"),
        (("set", "split", "unassigned"), "assigned split"),
    ],
)
def test_source_dataset_rejects_noncanonical_assigned_frames(
    tmp_path: Path, change: tuple[Any, ...], message: str
) -> None:
    frame = _manifest_frame(tmp_path)
    if change[0] == "drop":
        frame = frame.drop(columns=change[1])
    else:
        frame.iloc[0, frame.columns.get_loc(change[1])] = change[2]

    with pytest.raises(DataIntegrityError, match=message):
        _dataset_type()(frame)


def test_source_dataset_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    frame = _manifest_frame(tmp_path)
    frame.iloc[1, frame.columns.get_loc("sample_id")] = "sample-real"

    with pytest.raises(DataIntegrityError, match="sample_id.*unique"):
        _dataset_type()(frame)


def test_source_dataset_applies_exif_orientation_before_rgb_decode(tmp_path: Path) -> None:
    pixels = np.zeros((2, 3, 3), dtype=np.uint8)
    pixels[0, 0] = (255, 0, 0)
    pixels[1, 2] = (0, 255, 0)
    source = Image.fromarray(pixels, mode="RGB")
    exif = Image.Exif()
    exif[274] = 6
    path = tmp_path / "oriented.png"
    source.save(path, exif=exif)
    frame = _manifest_frame(tmp_path).iloc[[0]].copy()
    frame.iloc[0, frame.columns.get_loc("path")] = path

    item = _dataset_type()(frame)[0]

    expected = source.transpose(Image.Transpose.ROTATE_270)
    assert item.image.size == (2, 3)
    assert np.array_equal(np.asarray(item.image), np.asarray(expected))


@pytest.mark.parametrize("kind", ["missing", "corrupt", "truncated", "directory"])
def test_source_dataset_wraps_all_decode_failures_with_sample_and_path_context(
    tmp_path: Path, kind: str
) -> None:
    path = tmp_path / f"{kind}.png"
    if kind == "corrupt":
        path.write_bytes(b"not an image")
    elif kind == "truncated":
        complete = tmp_path / "complete.png"
        Image.new("RGB", (20, 20), "purple").save(complete)
        payload = complete.read_bytes()
        path.write_bytes(payload[: len(payload) // 2])
    elif kind == "directory":
        path.mkdir()
    frame = _manifest_frame(tmp_path).iloc[[0]].copy()
    frame.iloc[0, frame.columns.get_loc("path")] = path

    with pytest.raises(ImageDecodeError) as caught:
        _dataset_type()(frame)[0]

    message = str(caught.value)
    assert "sample-real" in message
    assert path.name in message


def test_source_dataset_closes_file_eagerly_and_keeps_detached_pixels(tmp_path: Path) -> None:
    frame = _manifest_frame(tmp_path).iloc[[0]].copy()
    original_path = Path(frame.iloc[0]["path"])

    item = _dataset_type()(frame)[0]
    moved = original_path.with_name("moved.png")
    original_path.rename(moved)

    assert item.image.getpixel((0, 0)) == (80, 80, 80)
    assert moved.is_file()
