from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from prooflens.errors import MetricPartitionError


def test_temperature_scaling_does_not_increase_validation_nll() -> None:
    from prooflens.evaluation.calibration import fit_temperature

    logits = torch.tensor([4.0, 3.0, -4.0, -3.0])
    labels = torch.tensor([1.0, 0.0, 0.0, 1.0])

    scaler = fit_temperature(logits, labels)
    before = F.binary_cross_entropy_with_logits(logits, labels)
    after = F.binary_cross_entropy_with_logits(scaler(logits), labels)

    assert after <= before + 1e-6
    assert scaler.temperature.item() > 0
    assert not scaler.training


def test_temperature_scaler_uses_positive_exponential_parameterization() -> None:
    from prooflens.evaluation.calibration import TemperatureScaler

    scaler = TemperatureScaler()
    with torch.no_grad():
        scaler.log_temperature.fill_(np.log(2.0))

    assert scaler.temperature.item() == pytest.approx(2.0)
    assert scaler(torch.tensor([2.0])).item() == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("logits", "labels", "message"),
    [
        (torch.tensor([[1.0, 2.0]]), torch.tensor([0.0, 1.0]), "one-dimensional"),
        (torch.tensor([1.0]), torch.tensor([0.0, 1.0]), "same length"),
        (torch.tensor([1.0, 2.0]), torch.tensor([0.0, 2.0]), "binary"),
        (torch.tensor([1.0, float("nan")]), torch.tensor([0.0, 1.0]), "finite"),
    ],
)
def test_temperature_fitting_rejects_invalid_validation_tensors(
    logits: torch.Tensor, labels: torch.Tensor, message: str
) -> None:
    from prooflens.evaluation.calibration import fit_temperature

    with pytest.raises(MetricPartitionError, match=message):
        fit_temperature(logits, labels)


def test_operating_threshold_is_fitted_on_validation_scores() -> None:
    from prooflens.evaluation.calibration import (
        compute_threshold_metrics,
        select_operating_threshold,
    )

    scores = np.array([0.05, 0.20, 0.70, 0.95])
    labels = np.array([0, 0, 1, 1])

    threshold = select_operating_threshold(scores, labels)
    report = compute_threshold_metrics(scores, labels, threshold)

    assert 0.20 < threshold <= 0.70
    assert report.false_positives == 0
    assert report.false_negatives == 0
    assert report.accuracy == pytest.approx(1.0)


def test_threshold_tie_break_chooses_value_closest_to_half() -> None:
    from prooflens.evaluation.calibration import select_operating_threshold

    scores = np.array([0.1, 0.4, 0.6, 0.9])
    labels = np.array([0, 1, 0, 1])

    threshold = select_operating_threshold(scores, labels)

    assert threshold == pytest.approx(0.4)


def test_calibration_json_records_validation_provenance(tmp_path) -> None:
    from prooflens.evaluation.calibration import write_calibration

    path = write_calibration(
        temperature=1.75,
        threshold=0.42,
        validation_split_hash="a" * 64,
        path=tmp_path / "calibration.json",
        fitted_at="2026-08-29T12:34:56Z",
    )
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload == {
        "fitted_at": "2026-08-29T12:34:56Z",
        "temperature": 1.75,
        "threshold": 0.42,
        "validation_split_hash": "a" * 64,
    }


@pytest.mark.parametrize(
    ("scores", "labels", "message"),
    [
        (np.array([0.2, np.nan]), np.array([0, 1]), "finite"),
        (np.array([0.2, 0.8]), np.array([0, 0]), "both labels"),
        (np.array([0.2]), np.array([0, 1]), "same length"),
    ],
)
def test_threshold_selection_rejects_invalid_validation_arrays(
    scores: np.ndarray, labels: np.ndarray, message: str
) -> None:
    from prooflens.evaluation.calibration import select_operating_threshold

    with pytest.raises(MetricPartitionError, match=message):
        select_operating_threshold(scores, labels)
