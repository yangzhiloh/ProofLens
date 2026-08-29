from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional

from prooflens.models.types import DetectorOutput, LossBreakdown


@dataclass(frozen=True, slots=True)
class SurvivalLossWeights:
    clean_bce: float = 1.0
    transformed_bce: float = 1.0
    prediction_consistency: float = 0.25
    feature_consistency: float = 0.10


DEFAULT_SURVIVAL_LOSS_WEIGHTS = SurvivalLossWeights()


def correct_margin(logits: Tensor, labels: Tensor) -> Tensor:
    if logits.shape != labels.shape:
        raise ValueError("logits and labels must have the same batch shape")
    return torch.where(labels == 1, logits, -logits)


def compute_survival_loss(
    clean: DetectorOutput,
    transformed: DetectorOutput,
    labels: Tensor,
    weights: SurvivalLossWeights = DEFAULT_SURVIVAL_LOSS_WEIGHTS,
) -> LossBreakdown:
    _validate_shapes(clean, transformed, labels)
    clean_bce = functional.binary_cross_entropy_with_logits(clean.logits, labels)
    transformed_bce = functional.binary_cross_entropy_with_logits(
        transformed.logits, labels
    )
    prediction_consistency = functional.mse_loss(clean.logits, transformed.logits)
    feature_consistency = (
        1
        - functional.cosine_similarity(clean.features, transformed.features, dim=1)
    ).mean()
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


def _validate_shapes(
    clean: DetectorOutput, transformed: DetectorOutput, labels: Tensor
) -> None:
    if labels.ndim != 1 or clean.logits.shape != labels.shape:
        raise ValueError("clean logits and labels must share a one-dimensional batch")
    if transformed.logits.shape != labels.shape:
        raise ValueError("transformed logits must match the label batch")
    if clean.features.ndim != 2 or transformed.features.shape != clean.features.shape:
        raise ValueError("clean and transformed feature batches must have matching shapes")
    if clean.features.shape[0] != labels.shape[0]:
        raise ValueError("feature batch must match the label batch")
