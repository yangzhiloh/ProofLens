"""Backend-neutral calibrated inference and transformation stability analysis."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Callable, Protocol

from PIL import Image

from prooflens.data.transforms import TransformSpec, apply_transform
from prooflens.errors import UserInputError
from prooflens.models.types import Prediction, StabilityResult


class LogitBackend(Protocol):
    model_version: str
    predict_logit: Callable[[Image.Image], float]


class InferenceService:
    def __init__(self, backend: LogitBackend, temperature: float = 1.0) -> None:
        if not callable(getattr(backend, "predict_logit", None)):
            raise TypeError("backend must provide predict_logit(image)")
        if temperature <= 0 or not math.isfinite(float(temperature)):
            raise ValueError("temperature must be positive")
        self.backend = backend
        self.temperature = float(temperature)

    @classmethod
    def from_calibration(cls, backend: LogitBackend, path: Path | None = None) -> "InferenceService":
        temperature = 1.0
        if path is not None:
            import json

            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            temperature = float(payload.get("temperature", 1.0))
        return cls(backend, temperature=temperature)

    def predict(self, image: Image.Image) -> Prediction:
        normalized = _validated_image(image)
        started = time.perf_counter()
        logit = float(self.backend.predict_logit(normalized))
        if not math.isfinite(logit):
            raise UserInputError("model returned a non-finite prediction")
        probability_ai = _sigmoid(logit / self.temperature)
        probability_real = 1.0 - probability_ai
        return Prediction(
            probability_ai=probability_ai,
            probability_real=probability_real,
            confidence=max(probability_ai, probability_real),
            logit=logit,
            model_version=str(getattr(self.backend, "model_version", "unknown")),
            inference_ms=(time.perf_counter() - started) * 1000.0,
        )

    def compare_transform(
        self, image: Image.Image, spec: TransformSpec, seed: int
    ) -> StabilityResult:
        normalized = _validated_image(image)
        clean = self.predict(normalized)
        transformed = self.predict(apply_transform(normalized, spec, seed))
        return StabilityResult(
            condition_id=spec.condition_id,
            clean=clean,
            transformed=transformed,
            absolute_change=abs(transformed.probability_ai - clean.probability_ai),
        )


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-min(value, 709.0))
        return 1.0 / (1.0 + z)
    z = math.exp(max(value, -709.0))
    return z / (1.0 + z)


def _validated_image(image: object) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise UserInputError("inference requires a PIL image")
    try:
        if image.width < 1 or image.height < 1:
            raise ValueError("empty image")
        converted = image.convert("RGB")
        converted.load()
        return converted.copy()
    except (OSError, TypeError, ValueError) as error:
        raise UserInputError("inference requires a decodable image") from error
