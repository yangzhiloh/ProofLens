"""Small, serializable value objects shared across model subsystems."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class DetectorOutput:
    logits: Tensor
    features: Tensor


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: Tensor
    clean_bce: Tensor
    transformed_bce: Tensor
    prediction_consistency: Tensor
    feature_consistency: Tensor


@dataclass(frozen=True, slots=True)
class Prediction:
    probability_ai: float
    probability_real: float
    confidence: float
    logit: float
    model_version: str
    inference_ms: float


@dataclass(frozen=True, slots=True)
class StabilityResult:
    condition_id: str
    clean: Prediction
    transformed: Prediction
    absolute_change: float
