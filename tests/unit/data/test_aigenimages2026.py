from __future__ import annotations

import csv
from pathlib import Path

import pytest
from PIL import Image

from prooflens.errors import DataIntegrityError


def _write_fixture(root: Path) -> None:
    (root / "0_real").mkdir(parents=True)
    (root / "1_fake" / "flux").mkdir(parents=True)
    Image.new("RGB", (12, 10), "white").save(root / "0_real" / "real.jpg")
    Image.new("RGB", (8, 6), "black").save(
        root / "1_fake" / "flux" / "fake.png"
    )
    with (root / "eval_real_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("fake", "real", "similarity"))
        writer.writeheader()
        writer.writerow({"fake": "fake.png", "real": "real.jpg", "similarity": "0.8"})


def test_adapter_preserves_pairs_generator_and_licence(tmp_path: Path) -> None:
    from prooflens.data.adapters.aigenimages2026 import AIGenImages2026Adapter

    _write_fixture(tmp_path)
    records = list(AIGenImages2026Adapter(tmp_path, version="pinned").scan())

    assert [record.label for record in records] == [0, 1]
    assert records[0].source_group_id == records[1].source_group_id
    assert records[1].generator_family == "flux"
    assert {record.dataset_name for record in records} == {"aigenimages2026"}
    assert {record.licence_identifier for record in records} == {"CC-BY-4.0"}
    assert (records[0].width, records[0].height) == (12, 10)
    assert (records[1].width, records[1].height) == (8, 6)


def test_adapter_rejects_missing_pair_image(tmp_path: Path) -> None:
    from prooflens.data.adapters.aigenimages2026 import AIGenImages2026Adapter

    _write_fixture(tmp_path)
    (tmp_path / "1_fake" / "flux" / "fake.png").unlink()
    Image.new("RGB", (4, 4), "red").save(
        tmp_path / "1_fake" / "flux" / "unreferenced.png"
    )

    with pytest.raises(DataIntegrityError, match="missing image"):
        list(AIGenImages2026Adapter(tmp_path, version="pinned").scan())


def test_adapter_uses_unique_normalized_match_for_archive_encoding(tmp_path: Path) -> None:
    from prooflens.data.adapters.aigenimages2026 import AIGenImages2026Adapter

    _write_fixture(tmp_path)
    original = tmp_path / "1_fake" / "flux" / "fake.png"
    damaged = original.with_name("fake-¦.png")
    original.rename(damaged)
    pairs = tmp_path / "eval_real_pairs.csv"
    pairs.write_text(
        "fake,real,similarity\nfake°.png,real.jpg,0.8\n", encoding="utf-8"
    )

    records = list(AIGenImages2026Adapter(tmp_path, version="pinned").scan())

    assert records[1].path == damaged.resolve()
