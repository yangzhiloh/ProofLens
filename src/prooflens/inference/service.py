from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image

from prooflens.data.transforms import TransformSpec, apply_transform
from prooflens.errors import DataIntegrityError, UserInputError
from prooflens.inference.preprocess import PREPROCESSING_VERSION


class LogitBackend(Protocol):
    """Structural interface shared by PyTorch and ONNX inference backends."""

    model_version: str
    preprocessing_version: str

    def predict_logit(self, image: Image.Image) -> float: ...


@dataclass(frozen=True, slots=True)
class Prediction:
    probability_ai: float
    probability_real: float
    confidence: float
    logit: float
    model_version: str
    preprocessing_version: str
    inference_ms: float


@dataclass(frozen=True, slots=True)
class StabilityResult:
    condition_id: str
    clean: Prediction
    transformed: Prediction
    absolute_change: float


class InferenceService:
    """Apply shared calibration and stability analysis over any logit backend."""

    def __init__(
        self,
        backend: LogitBackend,
        temperature: float,
        *,
        operating_threshold: float = 0.5,
    ) -> None:
        self.temperature = _positive_temperature(temperature)
        self.operating_threshold = _probability(operating_threshold, "operating threshold")
        self.backend = _validated_backend(backend)

    @classmethod
    def from_calibration(
        cls, backend: LogitBackend, calibration_path: Path
    ) -> InferenceService:
        """Construct a service from a validated Task 11 calibration artifact."""

        path = Path(calibration_path)
        if not path.is_file():
            raise DataIntegrityError(f"calibration file does not exist: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise DataIntegrityError(f"calibration file is unreadable: {path}") from error
        if not isinstance(payload, dict):
            raise DataIntegrityError("calibration artifact must contain a JSON object")
        missing = {"temperature", "threshold"}.difference(payload)
        if missing:
            raise DataIntegrityError(
                "calibration artifact is missing fields: " + ", ".join(sorted(missing))
            )
        try:
            return cls(
                backend,
                temperature=payload["temperature"],
                operating_threshold=payload["threshold"],
            )
        except (TypeError, ValueError) as error:
            raise DataIntegrityError("calibration artifact contains invalid values") from error

    def predict(self, image: Image.Image) -> Prediction:
        """Return calibrated probabilities and provenance for one image."""

        rgb = _validated_rgb_copy(image)
        started = time.perf_counter()
        logit = self.backend.predict_logit(rgb)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        numeric_logit = _finite_logit(logit)
        probability_ai = _sigmoid(numeric_logit / self.temperature)
        probability_real = 1.0 - probability_ai
        model_version = _backend_text(self.backend, "model_version")
        preprocessing_version = getattr(
            self.backend, "preprocessing_version", PREPROCESSING_VERSION
        )
        if not isinstance(preprocessing_version, str) or not preprocessing_version.strip():
            raise DataIntegrityError("backend preprocessing_version must be a nonempty string")
        return Prediction(
            probability_ai=probability_ai,
            probability_real=probability_real,
            confidence=max(probability_ai, probability_real),
            logit=numeric_logit,
            model_version=model_version,
            preprocessing_version=preprocessing_version,
            inference_ms=max(0.0, elapsed_ms),
        )

    def compare_transform(
        self, image: Image.Image, spec: TransformSpec, seed: int
    ) -> StabilityResult:
        """Compare calibrated AI probability before and after one canonical transform."""

        source = _validated_rgb_copy(image)
        clean = self.predict(source)
        transformed_image = apply_transform(source, spec, seed)
        transformed = self.predict(transformed_image)
        return StabilityResult(
            condition_id=spec.condition_id,
            clean=clean,
            transformed=transformed,
            absolute_change=abs(transformed.probability_ai - clean.probability_ai),
        )


def _validated_backend(backend: object) -> LogitBackend:
    if not callable(getattr(backend, "predict_logit", None)):
        raise TypeError("backend must provide a callable predict_logit method")
    _backend_text(backend, "model_version")
    return backend  # type: ignore[return-value]


def _backend_text(backend: object, field: str) -> str:
    value = getattr(backend, field, None)
    if not isinstance(value, str) or not value.strip():
        raise DataIntegrityError(f"backend {field} must be a nonempty string")
    return value


def _validated_rgb_copy(image: object) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise UserInputError("inference input must be a PIL image")
    try:
        image.load()
        return image.convert("RGB").copy()
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise UserInputError("inference input must be a decodable PIL image") from error


def _finite_logit(logit: object) -> float:
    if not isinstance(logit, (int, float)) or isinstance(logit, bool):
        raise DataIntegrityError("backend must return one finite numeric logit")
    numeric = float(logit)
    if not math.isfinite(numeric):
        raise DataIntegrityError("backend must return one finite numeric logit")
    return numeric


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _positive_temperature(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError("temperature must be a finite positive number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError("temperature must be a finite positive number")
    return numeric


def _probability(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{field} must be a finite probability")
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field} must be a finite probability")
    return numeric
