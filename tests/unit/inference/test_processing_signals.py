from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

from prooflens.inference.processing_signals import assess_processing


def test_assessment_reports_jpeg_encoding() -> None:
    buffer = BytesIO()
    Image.new("RGB", (512, 512), (80, 120, 160)).save(buffer, format="JPEG", quality=70)
    buffer.seek(0)
    encoded = Image.open(buffer)

    assessment = assess_processing(encoded)

    assert assessment.detected
    assert "JPEG encoding or compression" in assessment.likely_transformations
    assert any("JPEG encoded" in item for item in assessment.evidence)


def test_assessment_reports_low_resolution_without_claiming_proof() -> None:
    assessment = assess_processing(Image.new("RGB", (128, 96), (120, 120, 120)))

    assert assessment.detected
    assert "Low resolution; prior rescaling is possible" in assessment.likely_transformations
    assert assessment.confidence == "moderate"
    assert "not edit-history proof" in assessment.caveat


def test_assessment_reports_strong_blur() -> None:
    grid = np.indices((256, 256)).sum(axis=0) // 8 % 2
    checkerboard = Image.fromarray(np.uint8(grid * 255), mode="L").convert("RGB")
    blurred = checkerboard.filter(ImageFilter.GaussianBlur(radius=8.0))

    assessment = assess_processing(blurred)

    assert "Strong blur or loss of detail" in assessment.likely_transformations
