from PIL import Image
import pytest

from prooflens.data.transforms import get_spec
from prooflens.inference.service import InferenceService


class FakeBackend:
    model_version = "fake-1"

    def predict_logit(self, image: Image.Image) -> float:
        return 2.0


def test_prediction_probabilities_sum_to_one() -> None:
    service = InferenceService(FakeBackend(), temperature=2.0)
    prediction = service.predict(Image.new("RGB", (8, 8)))
    assert prediction.probability_ai + prediction.probability_real == pytest.approx(1.0)
    assert prediction.confidence == pytest.approx(max(prediction.probability_ai, prediction.probability_real))


def test_stability_reports_absolute_score_change() -> None:
    service = InferenceService(FakeBackend(), temperature=1.0)
    result = service.compare_transform(Image.new("RGB", (8, 8)), get_spec("jpeg_q30"), seed=17)
    assert result.condition_id == "jpeg_q30"
    assert result.absolute_change == pytest.approx(
        abs(result.transformed.probability_ai - result.clean.probability_ai)
    )
