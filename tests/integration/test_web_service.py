from __future__ import annotations

import pytest
from PIL import Image

from prooflens.inference.service import InferenceService


class FakeBackend:
    model_version = "fake-v1"
    preprocessing_version = "fake-224-v1"

    def predict_logit(self, image: Image.Image) -> float:
        return 0.25


def _service() -> InferenceService:
    return InferenceService(FakeBackend(), temperature=1.0)


def test_analyze_upload_returns_clean_and_transformed_results() -> None:
    from prooflens.web.app import analyze_upload

    image = Image.new("RGB", (16, 12), color=(120, 80, 40))

    result = analyze_upload(image, "jpeg_q30", _service())

    assert result.clean_image.size == (16, 12)
    assert result.transformed_image.size == (16, 12)
    assert result.summary["condition"] == "jpeg_q30"
    assert result.summary["clean"]["probability_ai"] == pytest.approx(0.5621765)


def test_analyze_upload_rejects_missing_image() -> None:
    from prooflens.errors import UserInputError
    from prooflens.web.app import analyze_upload

    with pytest.raises(UserInputError, match="Upload an image"):
        analyze_upload(None, "jpeg_q30", _service())
