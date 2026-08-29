"""Model definitions used by ProofLens."""

from prooflens.models.detector import DinoDetector
from prooflens.models.types import DetectorOutput, LossBreakdown, Prediction, StabilityResult

__all__ = [
    "DetectorOutput",
    "DinoDetector",
    "LossBreakdown",
    "Prediction",
    "StabilityResult",
]
