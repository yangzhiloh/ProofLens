from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from prooflens.errors import UserInputError


def _detail_image() -> Image.Image:
    y, x = np.indices((71, 103))
    pixels = np.stack(
        ((x * 17 + y * 3) % 256, (x * 5 + y * 19) % 256, (x * 13 + y * 7) % 256),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _png_bytes(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_stress_registry_has_the_four_secondary_conditions() -> None:
    from prooflens.data.stress_transforms import stress_specs

    assert tuple(spec.condition_id for spec in stress_specs()) == (
        "webp_q80",
        "webp_q50",
        "screenshot_1440",
        "screenshot_1080",
    )


@pytest.mark.parametrize("parameters", [{}, {"quality": 50}, None])
def test_stress_specs_reject_missing_or_mismatched_fixed_parameters(parameters: object) -> None:
    from prooflens.data.stress_transforms import StressTransformSpec

    with pytest.raises(UserInputError, match="stress transform spec"):
        StressTransformSpec("webp_q80", parameters)  # type: ignore[arg-type]


def test_stress_transforms_are_rgb_dimension_preserving_repeatable_and_visible() -> None:
    from prooflens.data.stress_transforms import apply_stress_transform, stress_specs

    image = _detail_image()
    source = np.asarray(image).copy()

    for spec in stress_specs():
        first = apply_stress_transform(image, spec, seed=73)
        second = apply_stress_transform(image, spec, seed=73)

        assert first.image.mode == "RGB"
        assert first.image.size == image.size
        assert _png_bytes(first.image) == _png_bytes(second.image)
        assert not np.array_equal(np.asarray(first.image), source)
        assert first.metadata["seed"] == 73
        assert first.metadata["condition_id"] == spec.condition_id
