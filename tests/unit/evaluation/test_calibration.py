import numpy as np
import pytest
import torch
from torch.nn import functional as F

from prooflens.evaluation.calibration import compute_threshold_metrics, fit_temperature, select_operating_threshold


def test_temperature_scaling_does_not_increase_validation_nll() -> None:
    logits = torch.tensor([4.0, 3.0, -4.0, -3.0])
    labels = torch.tensor([1.0, 0.0, 0.0, 1.0])
    scaler = fit_temperature(logits, labels)
    before = F.binary_cross_entropy_with_logits(logits, labels)
    after = F.binary_cross_entropy_with_logits(scaler(logits), labels)
    assert after <= before + 1e-6
    assert scaler.temperature.item() > 0


def test_operating_threshold_is_fitted_on_validation_scores() -> None:
    scores = np.array([0.05, 0.20, 0.70, 0.95])
    labels = np.array([0, 0, 1, 1])
    threshold = select_operating_threshold(scores, labels)
    report = compute_threshold_metrics(scores, labels, threshold)
    assert 0.20 < threshold <= 0.70
    assert report.false_positives == 0
    assert report.false_negatives == 0
