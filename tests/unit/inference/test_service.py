from __future__ import annotations

import json
import math

import pytest
from PIL import Image

from prooflens.errors import DataIntegrityError, UserInputError


class FakeBackend:
    model_version = "fake-v1"
    preprocessing_version = "fake-preprocess-v1"

    def __init__(self, logits: list[float] | None = None) -> None:
        self.logits = iter(logits or [2.0])
        self.received_modes: list[str] = []

    def predict_logit(self, image: Image.Image) -> float:
        self.received_modes.append(image.mode)
        return next(self.logits)


@pytest.fixture
def rgb_fixture() -> Image.Image:
    return Image.new("RGB", (40, 30), (120, 80, 20))


def test_prediction_probabilities_sum_to_one(rgb_fixture: Image.Image) -> None:
    from prooflens.inference.service import InferenceService

    service = InferenceService(FakeBackend(), temperature=2.0)

    prediction = service.predict(rgb_fixture)

    assert prediction.probability_ai + prediction.probability_real == pytest.approx(1.0)
    assert prediction.confidence == pytest.approx(
        max(prediction.probability_ai, prediction.probability_real)
    )
    assert prediction.probability_ai == pytest.approx(1.0 / (1.0 + math.exp(-1.0)))
    assert prediction.logit == pytest.approx(2.0)
    assert prediction.model_version == "fake-v1"
    assert prediction.preprocessing_version == "fake-preprocess-v1"
    assert prediction.inference_ms >= 0.0


def test_predict_converts_a_copy_to_rgb_without_mutating_source() -> None:
    from prooflens.inference.service import InferenceService

    backend = FakeBackend()
    service = InferenceService(backend, temperature=1.0)
    source = Image.new("RGBA", (12, 8), (10, 20, 30, 40))

    service.predict(source)

    assert backend.received_modes == ["RGB"]
    assert source.mode == "RGBA"
    assert source.getpixel((0, 0)) == (10, 20, 30, 40)


def test_stability_reports_absolute_score_change(rgb_fixture: Image.Image) -> None:
    from prooflens.data.transforms import get_spec
    from prooflens.inference.service import InferenceService

    service = InferenceService(FakeBackend([2.0, -1.0]), temperature=1.0)

    result = service.compare_transform(rgb_fixture, get_spec("jpeg_q30"), seed=17)

    assert result.condition_id == "jpeg_q30"
    assert result.absolute_change == pytest.approx(
        abs(result.transformed.probability_ai - result.clean.probability_ai)
    )


def test_service_loads_temperature_and_threshold_from_calibration_json(
    tmp_path, rgb_fixture: Image.Image
) -> None:
    from prooflens.inference.service import InferenceService

    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps(
            {
                "temperature": 2.5,
                "threshold": 0.43,
                "validation_split_hash": "abc",
                "fitted_at": "2026-08-29T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    service = InferenceService.from_calibration(FakeBackend(), calibration)

    assert service.temperature == pytest.approx(2.5)
    assert service.operating_threshold == pytest.approx(0.43)
    assert service.predict(rgb_fixture).probability_ai == pytest.approx(
        1.0 / (1.0 + math.exp(-2.0 / 2.5))
    )


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_service_rejects_invalid_temperature(temperature: float) -> None:
    from prooflens.inference.service import InferenceService

    with pytest.raises(ValueError, match="temperature"):
        InferenceService(FakeBackend(), temperature=temperature)


def test_service_rejects_a_non_image_input() -> None:
    from prooflens.inference.service import InferenceService

    with pytest.raises(UserInputError, match="PIL image"):
        InferenceService(FakeBackend(), temperature=1.0).predict(object())


@pytest.mark.parametrize("logit", [float("nan"), float("inf"), "not-a-logit"])
def test_service_rejects_malformed_backend_logits(logit) -> None:
    from prooflens.inference.service import InferenceService

    with pytest.raises(DataIntegrityError, match="finite numeric logit"):
        InferenceService(FakeBackend([logit]), temperature=1.0).predict(
            Image.new("RGB", (2, 2))
        )


def test_service_sigmoid_is_stable_for_extreme_logits(rgb_fixture: Image.Image) -> None:
    from prooflens.inference.service import InferenceService

    positive = InferenceService(FakeBackend([1000.0]), temperature=1.0).predict(rgb_fixture)
    negative = InferenceService(FakeBackend([-1000.0]), temperature=1.0).predict(rgb_fixture)

    assert positive.probability_ai == pytest.approx(1.0)
    assert negative.probability_ai == pytest.approx(0.0)
