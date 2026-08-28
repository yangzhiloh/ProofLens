from pathlib import Path

import imagehash
import pandas as pd
import pytest
from PIL import Image

from prooflens.data.hashing import enrich_hashes, perceptual_hash_file, sha256_file
from prooflens.errors import DataIntegrityError


def test_sha256_matches_identical_bytes_across_chunk_boundaries(tmp_path: Path) -> None:
    payload = b"exact-content" * 17
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(payload)
    second.write_bytes(payload)

    assert sha256_file(first, chunk_size=7) == sha256_file(second, chunk_size=11)
    assert sha256_file(first) == "1922e9f5867aa853c846851aab679e613adb7434e2b7f12e73aaa1c1ad4bf08d"


def test_sha256_rejects_nonpositive_chunk_size(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"content")

    with pytest.raises(DataIntegrityError, match="chunk_size"):
        sha256_file(path, chunk_size=0)


def test_perceptual_hash_applies_exif_orientation_before_hashing(tmp_path: Path) -> None:
    source = Image.new("RGB", (10, 6), "black")
    for x in range(3):
        for y in range(5):
            source.putpixel((x, y), (255, 40 + 20 * y, 0))
    source.putpixel((8, 1), (0, 255, 255))

    oriented_path = tmp_path / "oriented.png"
    exif = Image.Exif()
    exif[274] = 6
    source.save(oriented_path, exif=exif)

    physical_path = tmp_path / "physical.png"
    source.transpose(Image.Transpose.ROTATE_270).save(physical_path)

    assert perceptual_hash_file(oriented_path) == perceptual_hash_file(physical_path)
    assert perceptual_hash_file(oriented_path) == str(
        imagehash.phash(source.transpose(Image.Transpose.ROTATE_270).convert("RGB"))
    )


def test_enrich_hashes_returns_enriched_copy_without_mutating_input(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.new("L", (9, 7), 127).save(image_path)
    frame = pd.DataFrame(
        {
            "sample_id": ["sample"],
            "path": [image_path],
            "content_checksum": ["old-checksum"],
            "perceptual_hash": ["old-phash"],
        }
    )
    original = frame.copy(deep=True)

    enriched = enrich_hashes(frame)

    pd.testing.assert_frame_equal(frame, original)
    assert enriched is not frame
    assert enriched.loc[0, "content_checksum"] == sha256_file(image_path)
    assert enriched.loc[0, "perceptual_hash"] == perceptual_hash_file(image_path)
    assert len(enriched.loc[0, "content_checksum"]) == 64
    assert len(enriched.loc[0, "perceptual_hash"]) == 16


def test_enrich_hashes_rejects_missing_path_column_with_typed_error() -> None:
    with pytest.raises(DataIntegrityError, match="path"):
        enrich_hashes(pd.DataFrame({"sample_id": ["sample"]}))


@pytest.mark.parametrize("bad_path", [None, "", "   ", ["not", "a", "path"]])
def test_enrich_hashes_rejects_invalid_path_values_with_typed_error(
    bad_path: object,
) -> None:
    frame = pd.DataFrame({"sample_id": ["sample"], "path": [bad_path]})

    with pytest.raises(DataIntegrityError, match="nonempty local path"):
        enrich_hashes(frame)


def test_enrich_hashes_wraps_missing_file_failure_with_typed_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"
    frame = pd.DataFrame({"sample_id": ["sample"], "path": [missing]})

    with pytest.raises(DataIntegrityError, match=r"cannot hash file bytes.*missing\.png"):
        enrich_hashes(frame)


def test_enrich_hashes_wraps_undecodable_image_failure_with_typed_error(
    tmp_path: Path,
) -> None:
    unreadable = tmp_path / "unreadable.png"
    unreadable.write_bytes(b"not an image")
    frame = pd.DataFrame({"sample_id": ["sample"], "path": [unreadable]})

    with pytest.raises(
        DataIntegrityError,
        match=r"cannot perceptually hash image.*unreadable\.png",
    ):
        enrich_hashes(frame)
