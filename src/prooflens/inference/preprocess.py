from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np
import torch
from PIL import Image
from torch import Tensor

from prooflens.errors import DataIntegrityError

DINO_MODEL_ID = "facebook/dinov2-base"
PREPROCESSING_VERSION = "dinov2-base-224-v1"
FIXTURE_PREPROCESSING_VERSION = "fixture-rgb-224-v1"


class ImageProcessor(Protocol):
    def __call__(
        self, *, images: Sequence[Image.Image], return_tensors: str
    ) -> Mapping[str, Any]: ...


def create_dinov2_processor() -> ImageProcessor:
    """Explicitly load the committed DINOv2 processor on production request."""

    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(DINO_MODEL_ID)


class FixtureImageProcessor:
    """Deterministic RGB scaling used only by the miniature demo workflow."""

    def __call__(self, *, images: Sequence[Image.Image], return_tensors: str):
        values = np.stack(
            [
                np.asarray(image.resize((224, 224)), dtype=np.float32).transpose(2, 0, 1)
                / 255.0
                for image in images
            ]
        ).astype(np.float32)
        if return_tensors == "np":
            return {"pixel_values": values}
        if return_tensors == "pt":
            return {"pixel_values": torch.from_numpy(values)}
        raise ValueError("fixture processor supports only 'pt' and 'np' tensors")


def create_fixture_processor() -> ImageProcessor:
    """Create the offline processor paired with fixture-demo exports."""

    return FixtureImageProcessor()


def preprocess_images(
    images: Sequence[Image.Image], *, processor: ImageProcessor
) -> Tensor:
    """Run the injected processor once and enforce the shared pixel tensor contract."""

    copied = tuple(_validated_image(image, position) for position, image in enumerate(images))
    if not copied:
        raise DataIntegrityError("preprocessing requires a nonempty image sequence")
    output = processor(images=copied, return_tensors="pt")
    if not isinstance(output, Mapping) or "pixel_values" not in output:
        raise DataIntegrityError("processor output must contain pixel_values")
    pixel_values = output["pixel_values"]
    if not isinstance(pixel_values, Tensor):
        raise DataIntegrityError("processor pixel_values must be a torch tensor")
    if pixel_values.ndim != 4:
        raise DataIntegrityError("processor pixel_values must have rank 4")
    if pixel_values.shape[0] != len(copied):
        raise DataIntegrityError(
            "processor pixel_values batch dimension must match the input batch"
        )
    if tuple(pixel_values.shape[1:]) != (3, 224, 224):
        raise DataIntegrityError(
            "processor pixel_values must have shape [batch, 3, 224, 224]"
        )
    if not pixel_values.is_floating_point():
        raise DataIntegrityError("processor pixel_values must have a floating dtype")
    if not torch.isfinite(pixel_values).all().item():
        raise DataIntegrityError("processor pixel_values must contain only finite values")
    return pixel_values.to(dtype=torch.float32)


def _validated_image(image: object, position: int) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise DataIntegrityError(
            f"preprocessing image at position {position} must be a PIL image"
        )
    try:
        image.load()
        return image.copy()
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise DataIntegrityError(
            f"preprocessing image at position {position} must be decodable"
        ) from error
