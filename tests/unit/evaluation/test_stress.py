from __future__ import annotations

import pandas as pd
import pytest
from PIL import Image

from prooflens.data.dataset import SourceItem
from prooflens.errors import MetricPartitionError


@pytest.fixture
def stress_predictions() -> pd.DataFrame:
    rows = []
    for condition_id, values in {
        "webp_q80": ((0, 0.10, 0.15), (1, 0.90, 0.82)),
        "webp_q50": ((0, 0.20, 0.22), (1, 0.80, 0.71)),
        "screenshot_1440": ((0, 0.15, 0.20), (1, 0.85, 0.76)),
        "screenshot_1080": ((0, 0.25, 0.29), (1, 0.75, 0.68)),
    }.items():
        for label, clean_score, score in values:
            rows.append(
                {
                    "sample_id": f"{condition_id}-{label}",
                    "label": label,
                    "logit": 0.0,
                    "score": score,
                    "clean_score": clean_score,
                    "split": "validation",
                    "generator_family": "real" if label == 0 else "generator-a",
                    "condition_id": condition_id,
                    "checkpoint_id": "best",
                    "transform_metadata": '{"seed": 17}',
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def stress_report(stress_predictions: pd.DataFrame) -> dict[str, object]:
    from prooflens.evaluation.stress import compute_stress_metrics

    return compute_stress_metrics(stress_predictions)


def test_stress_metrics_are_not_checkpoint_ranking_input(stress_report: dict[str, object]) -> None:
    assert "ranking" not in stress_report
    assert set(stress_report["conditions"]) == {
        "webp_q80",
        "webp_q50",
        "screenshot_1440",
        "screenshot_1080",
    }


def test_stress_metrics_report_condition_auc_and_probability_shift(
    stress_report: dict[str, object],
) -> None:
    conditions = stress_report["conditions"]
    assert conditions["webp_q80"]["auc"] == pytest.approx(1.0)
    assert conditions["webp_q80"]["probability_shift"]["mean"] == pytest.approx(-0.015)


def test_stress_predictions_cannot_be_converted_to_primary_candidates(
    stress_predictions: pd.DataFrame,
) -> None:
    from prooflens.evaluation.select import select_best
    from prooflens.evaluation.stress import compute_stress_metrics

    report = compute_stress_metrics(stress_predictions)

    with pytest.raises(TypeError, match="must be a Candidate"):
        select_best([report])


@pytest.mark.parametrize(
    ("checkpoint_id", "seed"),
    [("", 17), (" ", 17), ("best", -1), ("best", 1.5), ("best", True)],
)
def test_evaluate_stress_rejects_invalid_public_identity_inputs(
    checkpoint_id: object, seed: object
) -> None:
    from prooflens.evaluation.stress import evaluate_stress

    item = SourceItem(
        image=Image.new("RGB", (4, 3), "red"),
        label=1,
        sample_id="sample",
        dataset_name="fixture",
        generator_family="generator",
        source_group_id="group",
        split="validation",
        split_group_id="group",
    )

    with pytest.raises(MetricPartitionError, match="checkpoint_id|seed"):
        evaluate_stress([item], object(), checkpoint_id=checkpoint_id, seed=seed)  # type: ignore[arg-type]
