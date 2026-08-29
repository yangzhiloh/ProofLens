from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def metric_report():
    from prooflens.evaluation.metrics import MetricReport

    return MetricReport(
        clean_auc=0.95,
        condition_auc={
            "blur_s0.5": 0.81,
            "center_crop_80": 0.82,
            "color_jitter_20": 0.83,
            "jpeg_q90": 0.84,
            "noise_s0.02": 0.85,
            "resize_x0.5": 0.86,
        },
        family_auc={
            "jpeg": 0.84,
            "blur": 0.81,
            "resize": 0.86,
            "noise": 0.85,
            "color_jitter": 0.83,
            "center_crop": 0.82,
        },
        macro_robust_auc=0.835,
        pooled_robust_auc=0.834,
        worst_condition_auc=0.81,
        worst_family_auc=0.81,
        unseen_generator_auc=0.79,
        composite_score=0.8925,
        model_parameters=86_000_001,
        inference_ms_median=12.5,
    )


@pytest.fixture
def threshold_report():
    from prooflens.evaluation.calibration import ThresholdReport

    return ThresholdReport(
        threshold=0.47,
        accuracy=0.91,
        precision=0.90,
        recall=0.92,
        f1=0.91,
        false_positives=4,
        false_negatives=3,
    )


def test_markdown_table_contains_required_rows(metric_report, threshold_report, tmp_path) -> None:
    from prooflens.reporting.tables import write_robustness_markdown

    path = write_robustness_markdown(
        metric_report, tmp_path / "robustness.md", threshold_report
    )
    text = path.read_text(encoding="utf-8")

    for name in (
        "Clean",
        "JPEG",
        "Blur",
        "Resize",
        "Noise",
        "Color jitter",
        "Center crop",
        "Macro robust",
        "Pooled robust",
        "Worst family",
        "Worst condition",
        "Unseen generator",
    ):
        assert name in text
    for name in (
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "False positives",
        "False negatives",
        "Model parameters",
        "Median CPU inference time",
    ):
        assert name in text


def test_metric_artifacts_write_strict_json_csv_and_markdown(
    metric_report, threshold_report, tmp_path
) -> None:
    from prooflens.reporting.tables import write_metric_artifacts

    json_path, csv_path, markdown_path = write_metric_artifacts(
        metric_report, threshold_report, tmp_path
    )
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    csv = pd.read_csv(csv_path)

    assert payload["ranking"]["clean_auc"] == pytest.approx(0.95)
    assert payload["operating_point"]["threshold"] == pytest.approx(0.47)
    assert set(csv.columns) == {"partition", "roc_auc"}
    assert markdown_path.is_file()


def test_auc_plot_is_written(metric_report, tmp_path) -> None:
    from prooflens.reporting.plots import write_auc_plot

    path = write_auc_plot(metric_report, tmp_path / "auc.png")

    assert path.is_file()
    assert path.stat().st_size > 0
    with Image.open(path) as image:
        assert image.format == "PNG"


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "real-high-1",
                "label": 0,
                "score": 0.99,
                "split": "validation",
                "checkpoint_id": "checkpoint-a",
                "condition_id": "clean",
                "generator_family": "real",
            },
            {
                "sample_id": "real-high-2",
                "label": 0,
                "score": 0.90,
                "split": "validation",
                "checkpoint_id": "checkpoint-a",
                "condition_id": "clean",
                "generator_family": "real",
            },
            {
                "sample_id": "real-low",
                "label": 0,
                "score": 0.10,
                "split": "validation",
                "checkpoint_id": "checkpoint-a",
                "condition_id": "clean",
                "generator_family": "real",
            },
            {
                "sample_id": "fake-low-1",
                "label": 1,
                "score": 0.01,
                "split": "validation",
                "checkpoint_id": "checkpoint-a",
                "condition_id": "clean",
                "generator_family": "generator-a",
            },
            {
                "sample_id": "fake-low-2",
                "label": 1,
                "score": 0.11,
                "split": "validation",
                "checkpoint_id": "checkpoint-a",
                "condition_id": "clean",
                "generator_family": "generator-a",
            },
            {
                "sample_id": "fake-high",
                "label": 1,
                "score": 0.95,
                "split": "validation",
                "checkpoint_id": "checkpoint-a",
                "condition_id": "clean",
                "generator_family": "generator-a",
            },
            {
                "sample_id": "real-high-1",
                "label": 0,
                "score": 0.05,
                "split": "validation",
                "checkpoint_id": "checkpoint-a",
                "condition_id": "jpeg_q30",
                "generator_family": "real",
            },
            {
                "sample_id": "fake-low-1",
                "label": 1,
                "score": 0.96,
                "split": "validation",
                "checkpoint_id": "checkpoint-a",
                "condition_id": "blur_s2.0",
                "generator_family": "generator-a",
            },
        ]
    )


def test_gallery_selects_highest_confidence_errors_and_unique_instability_rows() -> None:
    from prooflens.reporting.gallery import select_error_cases

    selected = select_error_cases(_prediction_frame(), per_category=2)

    assert len(selected.false_positives) == 2
    assert selected.false_positives["score"].is_monotonic_decreasing
    assert selected.false_positives["sample_id"].tolist() == ["real-high-1", "real-high-2"]
    assert len(selected.false_negatives) == 2
    assert selected.false_negatives["score"].is_monotonic_increasing
    assert len(selected.unstable) == 2
    assert selected.unstable["absolute_change"].is_monotonic_decreasing
    assert not selected.unstable.duplicated(["sample_id", "condition_id"]).any()


def test_error_gallery_writes_thumbnails_and_escapes_captions(tmp_path) -> None:
    from prooflens.reporting.gallery import select_error_cases, write_error_gallery

    predictions = _prediction_frame().copy()
    predictions.loc[0, "sample_id"] = "<script>alert(1)</script>"
    predictions.loc[6, "sample_id"] = "<script>alert(1)</script>"
    manifest_rows = []
    for index, sample_id in enumerate(predictions["sample_id"].unique()):
        source = tmp_path / f"source-{index}.png"
        Image.new("RGB", (320, 180), (index * 20, 50, 100)).save(source)
        manifest_rows.append(
            {"sample_id": sample_id, "path": source, "dataset_name": "dataset<&>"}
        )
    cases = select_error_cases(predictions, per_category=2)

    path = write_error_gallery(cases, pd.DataFrame(manifest_rows), tmp_path / "gallery")
    text = path.read_text(encoding="utf-8")

    assert path.name == "error-gallery.html"
    assert "<script>alert(1)</script>" not in text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in text
    assert "dataset&lt;&amp;&gt;" in text
    thumbnails = list((tmp_path / "gallery" / "thumbnails").glob("*.jpg"))
    assert thumbnails
    for thumbnail in thumbnails:
        with Image.open(thumbnail) as image:
            assert image.size == (224, 224)


def test_error_selection_rejects_missing_columns() -> None:
    from prooflens.errors import MetricPartitionError
    from prooflens.reporting.gallery import select_error_cases

    with pytest.raises(MetricPartitionError, match="condition_id"):
        select_error_cases(pd.DataFrame({"sample_id": ["a"], "label": [0], "score": [0.9]}))


def test_gallery_rejects_duplicate_manifest_sample_ids(tmp_path) -> None:
    from prooflens.errors import DataIntegrityError
    from prooflens.reporting.gallery import ErrorCases, write_error_gallery

    empty = pd.DataFrame(columns=_prediction_frame().columns)
    manifest = pd.DataFrame(
        {
            "sample_id": ["duplicate", "duplicate"],
            "path": [Path("a.png"), Path("b.png")],
            "dataset_name": ["a", "b"],
        }
    )

    with pytest.raises(DataIntegrityError, match="duplicate"):
        write_error_gallery(ErrorCases(empty, empty, empty), manifest, tmp_path)
