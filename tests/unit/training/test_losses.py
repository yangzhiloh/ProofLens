from __future__ import annotations

import pytest
import torch
from torch.nn import functional

from prooflens.models.types import DetectorOutput


def _output(logits: torch.Tensor, features: torch.Tensor) -> DetectorOutput:
    return DetectorOutput(logits=logits, features=functional.normalize(features, dim=1))


def test_identical_views_have_zero_consistency_losses() -> None:
    from prooflens.training.losses import compute_survival_loss

    output = _output(
        torch.tensor([0.2, -0.3]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )

    result = compute_survival_loss(output, output, torch.tensor([1.0, 0.0]))

    assert result.prediction_consistency.item() == pytest.approx(0.0)
    assert result.feature_consistency.item() == pytest.approx(0.0)


def test_total_matches_approved_weights_and_backpropagates() -> None:
    from prooflens.training.losses import compute_survival_loss

    clean_logits = torch.tensor([0.4, -0.2], requires_grad=True)
    transformed_logits = torch.tensor([0.1, 0.3], requires_grad=True)
    clean = _output(clean_logits, torch.tensor([[1.0, 1.0], [1.0, 0.0]]))
    transformed = _output(
        transformed_logits, torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    )

    result = compute_survival_loss(clean, transformed, torch.tensor([1.0, 0.0]))
    expected = (
        result.clean_bce
        + result.transformed_bce
        + 0.25 * result.prediction_consistency
        + 0.10 * result.feature_consistency
    )
    result.total.backward()

    assert torch.allclose(result.total, expected)
    assert clean_logits.grad is not None
    assert transformed_logits.grad is not None


def test_loss_rejects_mismatched_batch_shapes() -> None:
    from prooflens.training.losses import compute_survival_loss

    clean = _output(torch.tensor([0.1, 0.2]), torch.ones(2, 3))
    transformed = _output(torch.tensor([0.1]), torch.ones(1, 3))

    with pytest.raises(ValueError, match="batch"):
        compute_survival_loss(clean, transformed, torch.tensor([1.0, 0.0]))
