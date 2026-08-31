from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from PIL import Image, ImageOps, UnidentifiedImageError

from prooflens.errors import UserInputError

_SUPPORTED_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)


class _Prediction(Protocol):
    probability_ai: float


class PredictionService(Protocol):
    def predict(self, image: Image.Image) -> _Prediction: ...


def write_directory_predictions(
    input_dir: Path,
    output_path: Path,
    service: PredictionService,
) -> Path:
    """Predict every supported image and write the portable challenge JSON format."""

    input_dir = Path(input_dir)
    output_path = Path(output_path)
    if not input_dir.is_dir():
        raise UserInputError(f"input directory does not exist or is not a directory: {input_dir}")
    image_paths = sorted(
        (
            path
            for path in input_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
        ),
        key=lambda path: path.relative_to(input_dir).as_posix(),
    )
    if not image_paths:
        raise UserInputError(f"input directory contains no supported images: {input_dir}")

    records: list[dict[str, str | float]] = []
    for image_path in image_paths:
        try:
            with Image.open(image_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                prediction = service.predict(image)
        except (OSError, UnidentifiedImageError) as error:
            raise UserInputError(f"could not decode image {image_path}: {error}") from error
        probability = float(prediction.probability_ai)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise UserInputError(f"model returned an invalid probability for {image_path}")
        records.append(
            {
                "image_path": image_path.relative_to(input_dir).as_posix(),
                "pred": probability,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(records, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output_path
