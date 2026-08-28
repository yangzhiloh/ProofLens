import json
from pathlib import Path

import pandas as pd

from prooflens.data.audit import audit_manifest, write_audit


def _audit_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [0, 0, 1, 1, 1],
            "dataset_name": ["sid", "sid", "wild", "wild", "wild"],
            "generator_family": ["authentic", "authentic", "flux", "sdxl", "sdxl"],
            "file_format": ["PNG", "PNG", "JPEG", "JPEG", "PNG"],
            "width": [100, 200, 300, 400, 500],
            "height": [50, 100, 150, 200, 250],
            "content_checksum": ["a", "b", "c", "c", ""],
            "optional_metadata": ["known", None, "", "known", "known"],
        }
    )


def test_audit_contains_all_required_distribution_and_duplicate_fields() -> None:
    report = audit_manifest(_audit_frame())

    assert report.row_count == 5
    assert report.class_counts == {0: 2, 1: 3}
    assert report.dataset_counts == {"wild": 3, "sid": 2}
    assert report.generator_counts == {"authentic": 2, "sdxl": 2, "flux": 1}
    assert report.dimension_quantiles == {
        "width": {"min": 100.0, "q25": 200.0, "median": 300.0, "q75": 400.0, "max": 500.0},
        "height": {"min": 50.0, "q25": 100.0, "median": 150.0, "q75": 200.0, "max": 250.0},
    }
    assert report.file_format_crosstab == {"JPEG": {0: 0, 1: 2}, "PNG": {0: 2, 1: 1}}
    assert report.missing_counts["optional_metadata"] == 2
    assert report.missing_counts["content_checksum"] == 1
    assert report.exact_duplicate_count == 1
    assert "dataset_name" in report.perfect_shortcuts


def test_perfect_shortcuts_require_meaningful_observed_categories() -> None:
    frame = pd.DataFrame(
        {
            "label": [0, 1, 0, 1],
            "dataset_name": [None, None, None, None],
            "generator_family": ["", "", "", ""],
            "file_format": ["PNG", "PNG", "PNG", "PNG"],
            "width": [1, 1, 1, 1],
            "height": [1, 1, 1, 1],
        }
    )

    report = audit_manifest(frame)

    assert report.perfect_shortcuts == ()
    assert report.exact_duplicate_count == 0
    assert report.missing_counts["content_checksum"] == 4


def test_empty_audit_is_safe_and_does_not_report_shortcuts() -> None:
    frame = pd.DataFrame(
        columns=["label", "dataset_name", "generator_family", "file_format", "width", "height"]
    )

    report = audit_manifest(frame)

    assert report.row_count == 0
    assert report.perfect_shortcuts == ()
    assert report.exact_duplicate_count == 0
    assert report.dimension_quantiles == {
        "width": {"min": None, "q25": None, "median": None, "q75": None, "max": None},
        "height": {"min": None, "q25": None, "median": None, "q75": None, "max": None},
    }


def test_write_audit_round_trips_json_and_required_markdown_sections(tmp_path: Path) -> None:
    report = audit_manifest(_audit_frame())

    json_path, markdown_path = write_audit(report, tmp_path / "report")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = markdown_path.read_text(encoding="utf-8")
    assert set(payload) == {
        "row_count",
        "class_counts",
        "dataset_counts",
        "generator_counts",
        "dimension_quantiles",
        "file_format_crosstab",
        "missing_counts",
        "exact_duplicate_count",
        "perfect_shortcuts",
    }
    assert payload["file_format_crosstab"]["JPEG"] == {"0": 0, "1": 2}
    for heading in (
        "Row count",
        "Class counts",
        "Dataset counts",
        "Generator counts",
        "Dimension quantiles",
        "File format by label",
        "Missing metadata",
        "Exact duplicates",
        "Perfect label shortcuts",
    ):
        assert heading in markdown
