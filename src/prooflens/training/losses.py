"""Differentiable clean/transformed survival losses."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from prooflens.models.types import DetectorOutput, LossBreakdown


@dataclass(frozen=True, slots=True)
class SurvivalLossWeights:
    clean_bce: float = 1.0
    transformed_bce: float = 1.0
    prediction_consistency: float = 0.25
    feature_consistency: float = 0.10

    def __post_init__(self) -> None:
        values = (
            self.clean_bce,
            self.transformed_bce,
            self.prediction_consistency,
            self.feature_consistency,
        )
        if any(value < 0 for value in values):
            raise ValueError("loss weights must be nonnegative")


def compute_survival_loss(
    clean: DetectorOutput,
    transformed: DetectorOutput,
    labels: Tensor,
    weights: SurvivalLossWeights = SurvivalLossWeights(),
) -> LossBreakdown:
    _validate_outputs(clean, transformed, labels)
    clean_bce = F.binary_cross_entropy_with_logits(clean.logits, labels)
    transformed_bce = F.binary_cross_entropy_with_logits(transformed.logits, labels)
    prediction_consistency = F.mse_loss(clean.logits, transformed.logits)
    feature_consistency = (1.0 - F.cosine_similarity(clean.features, transformed.features)).mean()
    total = (
        weights.clean_bce * clean_bce
        + weights.transformed_bce * transformed_bce
        + weights.prediction_consistency * prediction_consistency
        + weights.feature_consistency * feature_consistency
    )
    return LossBreakdown(
        total=total,
        clean_bce=clean_bce,
        transformed_bce=transformed_bce,
        prediction_consistency=prediction_consistency,
        feature_consistency=feature_consistency,
    )


def correct_margin(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have the same shape")
    return torch.where(labels == 1, logits, -logits)


def _validate_outputs(clean: DetectorOutput, transformed: DetectorOutput, labels: Tensor) -> None:
    if clean.logits.shape != transformed.logits.shape or clean.logits.shape != labels.shape:
        raise ValueError("clean logits, transformed logits, and labels must have matching shapes")
    if clean.features.shape != transformed.features.shape:
        raise ValueError("clean and transformed features must have matching shapes")
    if clean.features.ndim != 2 or clean.logits.ndim != 1 or labels.ndim != 1:
        raise ValueError("detector outputs must be batched tensors")
