from __future__ import annotations

import pytest
from PIL import Image

from prooflens.inference.service import InferenceService


class FakeBackend:
    model_version = "fake-v1"
    preprocessing_version = "fake-224-v1"

    def predict_logit(self, image: Image.Image) -> float:
        return 0.25


def _service(*, threshold: float = 0.5) -> InferenceService:
    return InferenceService(
        FakeBackend(),
        temperature=1.0,
        operating_threshold=threshold,
    )


def test_analyze_upload_returns_clean_and_transformed_results() -> None:
    from prooflens.web.app import analyze_upload

    image = Image.new("RGB", (16, 12), color=(120, 80, 40))

    result = analyze_upload(image, "jpeg_q30", _service())

    assert result.clean_image.size == (16, 12)
    assert result.transformed_image.size == (16, 12)
    assert result.summary["condition"] == "jpeg_q30"
    assert result.summary["clean"]["probability_ai"] == pytest.approx(0.5621765)
    assert result.summary["decision"] == {
        "operating_threshold": 0.5,
        "clean_signal": "ai-generated",
        "transformed_signal": "ai-generated",
        "verdict_changed": False,
        "stability_rating": "Stable under transformation",
    }
    outputs = result.as_outputs()
    assert "AI-generated signal" in outputs[4]
    assert "Stable under transformation" in outputs[5]
    assert "fake-v1" in outputs[6]


def test_analyze_upload_uses_calibrated_operating_threshold_for_verdict() -> None:
    from prooflens.web.app import analyze_upload

    result = analyze_upload(
        Image.new("RGB", (16, 12), color=(120, 80, 40)),
        "jpeg_q30",
        _service(threshold=0.60),
    )

    assert result.summary["decision"]["clean_signal"] == "authentic"
    assert "Authentic signal" in result.as_outputs()[4]


def test_analyze_upload_rejects_missing_image() -> None:
    from prooflens.errors import UserInputError
    from prooflens.web.app import analyze_upload

    with pytest.raises(UserInputError, match="Upload an image"):
        analyze_upload(None, "jpeg_q30", _service())
