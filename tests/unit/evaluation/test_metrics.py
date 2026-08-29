from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from prooflens.errors import MetricPartitionError

FAMILY_CONDITIONS = {
    "jpeg": ("jpeg_q90", "jpeg_q30"),
    "blur": ("blur_s0.5",),
    "resize": ("resize_x0.5",),
    "noise": ("noise_s0.02",),
    "color_jitter": ("color_jitter_20",),
    "center_crop": ("center_crop_80",),
}


def _rows_for_condition(
    condition_id: str,
    family: str,
    scores: tuple[float, float, float, float],
    *,
    split: str = "validation",
) -> list[dict[str, object]]:
    labels = (0, 0, 1, 1)
    return [
        {
            "sample_id": f"{split}-{condition_id}-{index}",
            "label": label,
            "logit": 0.0,
            "score": score,
            "split": split,
            "generator_family": "real" if label == 0 else "generator-a",
            "transform_family": family,
            "condition_id": condition_id,
            "checkpoint_id": "checkpoint-a",
        }
        for index, (label, score) in enumerate(zip(labels, scores, strict=True))
    ]


@pytest.fixture
def prediction_frame() -> pd.DataFrame:
    rows = _rows_for_condition("clean", "clean", (0.1, 0.2, 0.8, 0.9))
    condition_scores = {
        "jpeg_q90": (0.1, 0.4, 0.6, 0.9),
        "jpeg_q30": (0.7, 0.8, 0.2, 0.3),
        "blur_s0.5": (0.1, 0.2, 0.8, 0.9),
        "resize_x0.5": (0.1, 0.7, 0.3, 0.9),
        "noise_s0.02": (0.6, 0.7, 0.3, 0.4),
        "color_jitter_20": (0.1, 0.4, 0.6, 0.9),
        "center_crop_80": (0.2, 0.6, 0.4, 0.8),
    }
    for family, conditions in FAMILY_CONDITIONS.items():
        for condition in conditions:
            rows.extend(_rows_for_condition(condition, family, condition_scores[condition]))
    rows.extend(
        _rows_for_condition(
            "clean",
            "clean",
            (0.15, 0.25, 0.75, 0.85),
            split="generator_validation",
        )
    )
    return pd.DataFrame(rows)


def test_macro_robust_weights_families_equally(prediction_frame: pd.DataFrame) -> None:
    from prooflens.evaluation.metrics import compute_metrics

    report = compute_metrics(prediction_frame)

    expected = np.mean(
        [
            report.family_auc["jpeg"],
            report.family_auc["blur"],
            report.family_auc["resize"],
            report.family_auc["noise"],
            report.family_auc["color_jitter"],
            report.family_auc["center_crop"],
        ]
    )
    assert report.macro_robust_auc == pytest.approx(expected)
    assert report.family_auc["jpeg"] == pytest.approx(
        np.mean([report.condition_auc["jpeg_q90"], report.condition_auc["jpeg_q30"]])
    )


def test_report_contains_clean_pooled_worst_unseen_and_composite_metrics(
    prediction_frame: pd.DataFrame,
) -> None:
    from prooflens.evaluation.metrics import compute_condition_auc, compute_metrics

    report = compute_metrics(prediction_frame)
    robust = prediction_frame[
        (prediction_frame["split"] == "validation")
        & (prediction_frame["condition_id"] != "clean")
    ]

    assert report.clean_auc == pytest.approx(1.0)
    assert report.unseen_generator_auc == pytest.approx(1.0)
    assert report.pooled_robust_auc == pytest.approx(compute_condition_auc(robust))
    assert report.worst_condition_auc == min(report.condition_auc.values())
    assert report.worst_family_auc == min(report.family_auc.values())
    assert report.composite_score == pytest.approx(
        0.5 * report.clean_auc + 0.5 * report.macro_robust_auc
    )


def test_compute_metrics_uses_only_requested_partitions(prediction_frame: pd.DataFrame) -> None:
    from prooflens.evaluation.metrics import compute_metrics

    final_rows = prediction_frame.copy()
    final_rows["split"] = final_rows["split"].replace(
        {"validation": "test", "generator_validation": "generator_test"}
    )
    final_rows["score"] = 1.0 - final_rows["score"]
    combined = pd.concat([prediction_frame, final_rows], ignore_index=True)

    validation = compute_metrics(combined)
    final = compute_metrics(combined, evaluation_split="test", generator_split="generator_test")

    assert validation.clean_auc == pytest.approx(1.0)
    assert final.clean_auc == pytest.approx(0.0)
    assert validation.unseen_generator_auc == pytest.approx(1.0)
    assert final.unseen_generator_auc == pytest.approx(0.0)


def test_metric_rejects_single_class_condition() -> None:
    from prooflens.evaluation.metrics import compute_condition_auc

    with pytest.raises(MetricPartitionError, match="both labels"):
        compute_condition_auc(pd.DataFrame({"label": [1, 1], "score": [0.2, 0.8]}))


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"label": [0, 1]}), "score"),
        (pd.DataFrame({"label": [0, 1], "score": [0.2, math.nan]}), "finite"),
        (pd.DataFrame({"label": [0, 2], "score": [0.2, 0.8]}), "binary"),
    ],
)
def test_condition_auc_rejects_malformed_partitions(
    frame: pd.DataFrame, message: str
) -> None:
    from prooflens.evaluation.metrics import compute_condition_auc

    with pytest.raises(MetricPartitionError, match=message):
        compute_condition_auc(frame)


def test_compute_metrics_rejects_a_condition_mapped_to_multiple_families(
    prediction_frame: pd.DataFrame,
) -> None:
    from prooflens.evaluation.metrics import compute_metrics

    bad = prediction_frame.copy()
    row = bad.index[bad["condition_id"] == "jpeg_q90"][0]
    bad.loc[row, "transform_family"] = "blur"

    with pytest.raises(MetricPartitionError, match="exactly one transform family"):
        compute_metrics(bad)


def test_prediction_records_have_stable_schema_sigmoid_scores_and_atomic_parquet(
    tmp_path,
) -> None:
    from prooflens.evaluation.predict import (
        PREDICTION_COLUMNS,
        PredictionRecord,
        records_to_frame,
        write_predictions,
    )

    record = PredictionRecord.from_logit(
        sample_id="sample-1",
        label=1,
        logit=2.0,
        split="validation",
        generator_family="generator-a",
        transform_family="jpeg",
        condition_id="jpeg_q90",
        checkpoint_id="checkpoint-a",
    )
    frame = records_to_frame([record])
    output = tmp_path / "predictions.parquet"

    result = write_predictions([record], output)

    assert tuple(frame.columns) == PREDICTION_COLUMNS
    assert record.score == pytest.approx(1.0 / (1.0 + math.exp(-2.0)))
    assert result == output
    pd.testing.assert_frame_equal(pd.read_parquet(output), frame)
    assert not list(tmp_path.glob(".*.tmp"))


def test_prediction_record_rejects_an_inconsistent_sigmoid_score() -> None:
    from prooflens.evaluation.predict import PredictionRecord

    with pytest.raises(MetricPartitionError, match="sigmoid"):
        PredictionRecord(
            sample_id="sample-1",
            label=0,
            logit=0.0,
            score=0.9,
            split="validation",
            generator_family="real",
            transform_family="clean",
            condition_id="clean",
            checkpoint_id="checkpoint-a",
        )
