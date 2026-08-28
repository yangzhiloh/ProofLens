import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml
from PIL import Image

from prooflens.data.acquire import (
    SidAcquisitionConfig,
    acquire_sid_subset,
    hash_acquisition_config,
    load_primary_policy,
    select_balanced_binary_rows,
    validate_primary_manifest,
    validate_wildfake_root,
)
from prooflens.data.schema import MANIFEST_COLUMNS
from prooflens.errors import DataIntegrityError, UserInputError


class TinyStreamingDataset:
    dataset_revision = "observed-revision-987"

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __iter__(self):
        yield from self.rows


def _sid_row(image_id: str, label: int, color: str) -> dict[str, Any]:
    return {"img_id": image_id, "label": label, "image": Image.new("RGBA", (5, 4), color)}


def test_balanced_selector_stops_exactly_at_per_class_cap() -> None:
    rows = (
        {"img_id": f"{label}-{index}", "label": label}
        for index in range(10)
        for label in (0, 1, 2)
    )

    selected = list(select_balanced_binary_rows(rows, per_class=3))

    assert [row["label"] for row in selected].count(0) == 3
    assert [row["label"] for row in selected].count(1) == 3
    assert len(selected) == 6
    assert all(row["label"] in (0, 1) for row in selected)


def test_acquire_sid_subset_persists_rgb_images_manifest_and_metadata(tmp_path: Path) -> None:
    rows = [
        _sid_row("real/id", 0, "white"),
        _sid_row("tampered", 2, "red"),
        _sid_row("fake:id", 1, "black"),
        _sid_row("unused-real", 0, "blue"),
    ]

    def loader(dataset_id: str, **kwargs: object) -> TinyStreamingDataset:
        if dataset_id != "saberzl/SID_Set":
            raise AssertionError("unexpected dataset")
        if kwargs != {"split": "train", "streaming": True, "revision": "pinned-123"}:
            raise AssertionError("streaming contract changed")
        return TinyStreamingDataset(rows)

    output_root = tmp_path / "sid"
    summary = acquire_sid_subset(
        {"revision": "pinned-123", "per_class": 1},
        output_root,
        dataset_loader=loader,
    )

    frame = pd.read_parquet(summary.manifest_path)
    metadata = json.loads((output_root / "acquisition.json").read_text(encoding="utf-8"))
    assert tuple(frame.columns) == MANIFEST_COLUMNS
    assert set(frame["sample_id"]) == {"real/id", "fake:id"}
    assert frame["label"].value_counts().sort_index().to_dict() == {0: 1, 1: 1}
    assert all(Path(path).is_file() for path in frame["path"])
    assert all(Image.open(path).mode == "RGB" for path in frame["path"])
    assert summary.counts == {0: 1, 1: 1}
    assert summary.dataset_revision == "pinned-123"
    assert summary.observed_dataset_revision == "observed-revision-987"
    assert metadata == {
        "config_sha256": summary.config_sha256,
        "counts": {"0": 1, "1": 1},
        "dataset_id": "saberzl/SID_Set",
        "dataset_revision": "pinned-123",
        "licence_identifier": "CC-BY-4.0",
        "observed_dataset_revision": "observed-revision-987",
        "split": "train",
    }


def test_acquire_sid_subset_rejects_underfilled_stream_without_partial_output(
    tmp_path: Path,
) -> None:
    def loader(dataset_id: str, **kwargs: object) -> TinyStreamingDataset:
        return TinyStreamingDataset([_sid_row("only-real", 0, "white")])

    output_root = tmp_path / "underfilled"

    with pytest.raises(DataIntegrityError, match="underfilled.*label 1"):
        acquire_sid_subset(
            {"revision": "pinned-123", "per_class": 1},
            output_root,
            dataset_loader=loader,
        )

    assert not output_root.exists()


def test_acquire_sid_subset_refuses_to_overwrite_existing_root(tmp_path: Path) -> None:
    output_root = tmp_path / "sid"
    output_root.mkdir()
    marker = output_root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(UserInputError, match="already exists"):
        acquire_sid_subset({"revision": "pinned-123", "per_class": 1}, output_root)

    assert marker.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("images_directory", "/outside-images"),
        ("manifest_name", r"C:\outside\manifest.parquet"),
        ("images_directory", "../outside-images"),
        ("manifest_name", r"nested\..\..\manifest.parquet"),
        ("images_directory", ""),
        ("manifest_name", "   "),
    ],
)
def test_sid_acquisition_config_rejects_output_paths_that_can_escape_root(
    field: str, value: str
) -> None:
    with pytest.raises(UserInputError, match=field):
        SidAcquisitionConfig.from_value({field: value})


@pytest.mark.parametrize("field", ["images_directory", "manifest_name"])
@pytest.mark.parametrize(
    "value",
    [None, True, 7, 3.5, ["nested"], {"path": "nested"}, Path("nested")],
)
def test_sid_acquisition_config_rejects_non_string_output_paths(
    field: str, value: object
) -> None:
    with pytest.raises(UserInputError, match=f"{field}.*string"):
        SidAcquisitionConfig.from_value({field: value})


@pytest.mark.parametrize(
    "overrides",
    [
        {"manifest_name": "acquisition.json"},
        {"manifest_name": ".\\AcQuIsItIoN.JsOn"},
        {"images_directory": "acquisition.json"},
        {"images_directory": "ACQUISITION.JSON"},
        {"images_directory": "images", "manifest_name": "images/manifest.parquet"},
        {"images_directory": "IMAGES", "manifest_name": r"images\manifest.parquet"},
        {
            "images_directory": "reports/manifest.parquet/images",
            "manifest_name": "reports/manifest.parquet",
        },
        {"images_directory": "acquisition.json/images"},
        {"manifest_name": "acquisition.json/manifest.parquet"},
    ],
)
def test_sid_acquisition_config_rejects_colliding_output_layouts(
    overrides: dict[str, str]
) -> None:
    with pytest.raises(UserInputError, match="output paths overlap"):
        SidAcquisitionConfig.from_value(overrides)


def test_config_hash_is_deterministic_across_mapping_order() -> None:
    first = {"revision": "abc", "per_class": 2, "split": "train"}
    second = {"split": "train", "per_class": 2, "revision": "abc"}

    assert hash_acquisition_config(first) == hash_acquisition_config(second)
    assert len(hash_acquisition_config(first)) == 64


def test_missing_wildfake_root_points_to_official_repository_and_modelscope(
    tmp_path: Path,
) -> None:
    with pytest.raises(DataIntegrityError) as caught:
        validate_wildfake_root(tmp_path / "missing")

    message = str(caught.value)
    assert "https://github.com/hy-zpg/AIGC-Image-Detection-Dataset" in message
    assert "https://modelscope.cn/datasets/hy2628982280/WildFake/summary" in message
    assert "manual" in message.lower()


def test_validate_wildfake_root_accepts_a_nonempty_export(tmp_path: Path) -> None:
    export = tmp_path / "wildfake"
    (export / "real").mkdir(parents=True)
    (export / "real" / "one.jpg").write_bytes(b"fixture")

    assert validate_wildfake_root(export) == export


def test_primary_policy_has_top_level_three_family_gate_and_excludes_cifake() -> None:
    policy_path = Path("configs/data/primary.yaml")
    raw = yaml.safe_load(policy_path.read_text(encoding="utf-8"))

    assert raw["minimum_generator_families"] == 3
    assert all("minimum_generator_families" not in source for source in raw["sources"])
    assert {source["name"] for source in raw["sources"]} == {"sid_set", "wildfake"}
    assert load_primary_policy(policy_path).minimum_generator_families == 3


def test_dataset_configs_pin_sid_defaults_and_keep_cifake_stress_only() -> None:
    sid = yaml.safe_load(Path("configs/data/sid_subset.yaml").read_text(encoding="utf-8"))
    cifake = yaml.safe_load(Path("configs/data/cifake.yaml").read_text(encoding="utf-8"))

    assert sid["per_class"] == 10_000
    assert len(sid["revision"]) == 40
    assert SidAcquisitionConfig().revision == sid["revision"]
    assert cifake["stress_only"] is True
    assert cifake["include_in_primary_training"] is False


def test_primary_policy_rejects_missing_label_and_too_few_fake_families() -> None:
    policy = load_primary_policy(Path("configs/data/primary.yaml"))
    missing_real = pd.DataFrame(
        {
            "label": [1, 1, 1],
            "dataset_name": ["wildfake"] * 3,
            "generator_family": ["flux", "sdxl", "dalle3"],
        }
    )
    with pytest.raises(DataIntegrityError, match="both labels"):
        validate_primary_manifest(missing_real, policy)

    two_families = pd.DataFrame(
        {
            "label": [0, 1, 1],
            "dataset_name": ["sid_set", "wildfake", "wildfake"],
            "generator_family": ["authentic", "flux", "sdxl"],
        }
    )
    with pytest.raises(DataIntegrityError, match="at least 3.*found 2"):
        validate_primary_manifest(two_families, policy)


@pytest.mark.parametrize("unidentified_name", [None, "", "   "])
def test_primary_policy_rejects_unidentified_dataset_names(
    unidentified_name: str | None,
) -> None:
    policy = load_primary_policy(Path("configs/data/primary.yaml"))
    frame = pd.DataFrame(
        {
            "label": [0, 1, 1, 1, 0],
            "dataset_name": [
                "sid_set",
                "wildfake",
                "wildfake",
                "wildfake",
                unidentified_name,
            ],
            "generator_family": ["authentic", "flux", "sdxl", "dalle3", "authentic"],
        }
    )

    with pytest.raises(DataIntegrityError, match="dataset_name.*missing or blank"):
        validate_primary_manifest(frame, policy)


def test_primary_policy_counts_generator_families_across_approved_sources() -> None:
    policy = load_primary_policy(Path("configs/data/primary.yaml"))
    frame = pd.DataFrame(
        {
            "label": [0, 1, 1, 1, 1],
            "dataset_name": ["sid_set", "sid_set", "wildfake", "wildfake", "wildfake"],
            "generator_family": ["authentic", "generated", "flux", "sdxl", "dalle3"],
        }
    )

    validate_primary_manifest(frame, policy)
