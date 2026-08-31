from __future__ import annotations

import pytest
from PIL import Image

from prooflens.inference.service import InferenceService


class FakeBackend:
    model_version = "fake-v1"
    preprocessing_version = "fake-224-v1"

    def predict_logit(self, image: Image.Image) -> float:
        return 0.25


def _service(*, operating_threshold: float = 0.5) -> InferenceService:
    return InferenceService(
        FakeBackend(),
        temperature=1.0,
        operating_threshold=operating_threshold,
    )


def test_analyze_upload_returns_clean_and_transformed_results() -> None:
    from prooflens.web.app import analyze_upload

    image = Image.new("RGB", (16, 12), color=(120, 80, 40))

    result = analyze_upload(image, "jpeg_q30", _service(operating_threshold=0.6))

    assert result.clean_image.size == (16, 12)
    assert result.transformed_image.size == (16, 12)
    assert result.summary["condition"] == "jpeg_q30"
    assert result.summary["clean"]["probability_ai"] == pytest.approx(0.5621765)
    assert result.summary["transform_parameters"] == {"quality": 30, "subsampling": 2}
    assert result.summary["operating_threshold"] == pytest.approx(0.6)
    assert result.summary["model_version"] == "fake-v1"
    assert result.summary["preprocessing_version"] == "fake-224-v1"
    for prediction_name in ("clean", "transformed"):
        prediction = result.summary[prediction_name]
        assert prediction["probability_ai"] == pytest.approx(0.5621765)
        assert prediction["confidence"] == pytest.approx(0.5621765)
        assert prediction["inference_ms"] >= 0.0
        assert prediction["threshold_label"] == "Authentic"
    assert result.summary["decision"] == {
        "operating_threshold": 0.6,
        "clean_signal": "authentic",
        "transformed_signal": "authentic",
        "verdict_changed": False,
        "stability_rating": "Stable under transformation",
    }
    outputs = result.as_outputs()
    assert "Authentic signal" in outputs[4]
    assert "Stable under transformation" in outputs[5]
    assert "fake-v1" in outputs[6]


def test_analyze_upload_uses_calibrated_operating_threshold_for_verdict() -> None:
    from prooflens.web.app import analyze_upload

    result = analyze_upload(
        Image.new("RGB", (16, 12), color=(120, 80, 40)),
        "jpeg_q30",
        _service(operating_threshold=0.60),
    )

    assert result.summary["decision"]["clean_signal"] == "authentic"
    assert "Authentic signal" in result.as_outputs()[4]


def test_transformation_labels_are_readable_and_cover_every_canonical_condition() -> None:
    from prooflens.data.transforms import canonical_specs
    from prooflens.web.app import _transformation_label

    labels = {
        spec.condition_id: _transformation_label(spec.condition_id)
        for spec in canonical_specs()
    }

    assert labels["jpeg_q30"] == "JPEG compression — Heavy (quality 30)"
    assert labels["blur_s2.0"] == "Blur — Strong (strength 2.0)"
    assert labels["resize_x0.25"] == "Resize — Quarter resolution (25%)"
    assert labels["center_crop_80"] == "Center crop — Keep middle 80%"
    assert all(condition_id not in label for condition_id, label in labels.items())


def test_analyze_upload_rejects_missing_image() -> None:
    from prooflens.errors import UserInputError
    from prooflens.web.app import analyze_upload

    with pytest.raises(UserInputError, match="Upload an image"):
        analyze_upload(None, "jpeg_q30", _service())
