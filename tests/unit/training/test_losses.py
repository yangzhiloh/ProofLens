import pytest
import torch
from torch.nn import functional as F

from prooflens.models.types import DetectorOutput
from prooflens.training.losses import compute_survival_loss


def test_identical_views_have_zero_consistency_losses() -> None:
    output = DetectorOutput(
        logits=torch.tensor([0.2, -0.3]),
        features=F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), dim=1),
    )
    result = compute_survival_loss(output, output, torch.tensor([1.0, 0.0]))
    assert result.prediction_consistency.item() == pytest.approx(0.0)
    assert result.feature_consistency.item() == pytest.approx(0.0)


def test_total_matches_approved_weights() -> None:
    clean = DetectorOutput(torch.tensor([0.2, -0.3]), F.normalize(torch.eye(2), dim=1))
    transformed = DetectorOutput(torch.tensor([0.5, -0.1]), F.normalize(torch.tensor([[0.8, 0.2], [0.2, 0.8]]), dim=1))
    result = compute_survival_loss(clean, transformed, torch.tensor([1.0, 0.0]))
    expected = result.clean_bce + result.transformed_bce + 0.25 * result.prediction_consistency + 0.10 * result.feature_consistency
    assert torch.allclose(result.total, expected)
