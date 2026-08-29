from PIL import Image
import pytest

from prooflens.inference.service import InferenceService
from prooflens.web.app import analyze_upload


class FakeBackend:
    model_version = "fake"

    def predict_logit(self, image):
        return 0.25


def test_analyze_upload_returns_clean_and_transformed_results() -> None:
    image = Image.new("RGB", (16, 12), color=(120, 80, 40))
    result = analyze_upload(image, "jpeg_q30", InferenceService(FakeBackend()))
    assert result.clean_image.size == image.size
    assert result.transformed_image.size == image.size
    assert result.summary["condition"] == "jpeg_q30"
    assert "probability_ai" in result.summary["clean"]


def test_analyze_upload_rejects_missing_image() -> None:
    from prooflens.errors import UserInputError

    with pytest.raises(UserInputError, match="Upload an image"):
        analyze_upload(None, "jpeg_q30", InferenceService(FakeBackend()))
