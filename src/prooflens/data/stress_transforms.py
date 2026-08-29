"""Deterministic secondary redistribution transforms outside primary ranking."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from numbers import Integral
from types import MappingProxyType

from PIL import Image

from prooflens.errors import DataIntegrityError, UserInputError


@dataclass(frozen=True, slots=True)
class StressTransformSpec:
    """One immutable supplemental redistribution condition."""

    condition_id: str
    parameters: Mapping[str, int | str]

    def __post_init__(self) -> None:
        if self.condition_id not in {definition[0] for definition in _SPEC_DEFINITIONS}:
            raise UserInputError(f"unknown stress condition {self.condition_id!r}")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class StressTransformResult:
    """An image plus reproducibility metadata for a supplemental condition."""

    image: Image.Image
    metadata: Mapping[str, int | str]


_SPEC_DEFINITIONS: tuple[tuple[str, dict[str, int | str]], ...] = (
    ("webp_q80", {"codec": "webp", "quality": 80}),
    ("webp_q50", {"codec": "webp", "quality": 50}),
    (
        "screenshot_1440",
        {"canvas_width": 1440, "canvas_height": 900, "interpolation": "bicubic"},
    ),
    (
        "screenshot_1080",
        {"canvas_width": 1080, "canvas_height": 675, "interpolation": "bicubic"},
    ),
)
_STRESS_SPECS = tuple(StressTransformSpec(condition_id, parameters) for condition_id, parameters in _SPEC_DEFINITIONS)


def stress_specs() -> tuple[StressTransformSpec, ...]:
    """Return the fixed secondary conditions in report order."""

    return _STRESS_SPECS


def apply_stress_transform(
    image: Image.Image, spec: StressTransformSpec, *, seed: int
) -> StressTransformResult:
    """Apply a deterministic redistribution transform without mutating the source."""

    if not isinstance(image, Image.Image):
        raise UserInputError("stress transform image must be a PIL image")
    if not isinstance(spec, StressTransformSpec):
        raise UserInputError("stress transform spec must be a StressTransformSpec")
    if not isinstance(seed, Integral) or isinstance(seed, bool) or int(seed) < 0:
        raise UserInputError("stress transform seed must be a nonnegative integer")
    rgb = image.convert("RGB")
    if spec.condition_id.startswith("webp_"):
        transformed = _webp_round_trip(rgb, int(spec.parameters["quality"]))
        metadata = {**spec.parameters, "seed": int(seed), "condition_id": spec.condition_id}
    else:
        transformed = _screenshot_round_trip(
            rgb,
            int(spec.parameters["canvas_width"]),
            int(spec.parameters["canvas_height"]),
        )
        metadata = {
            **spec.parameters,
            "capture_codec": "png",
            "seed": int(seed),
            "condition_id": spec.condition_id,
        }
    if transformed.mode != "RGB" or transformed.size != rgb.size:
        raise DataIntegrityError("stress transform violated the RGB dimension contract")
    return StressTransformResult(transformed, MappingProxyType(metadata))


def _webp_round_trip(image: Image.Image, quality: int) -> Image.Image:
    encoded = BytesIO()
    image.save(encoded, format="WEBP", quality=quality, method=6)
    encoded.seek(0)
    with Image.open(encoded) as decoded:
        decoded.load()
        return decoded.convert("RGB").copy()


def _screenshot_round_trip(image: Image.Image, canvas_width: int, canvas_height: int) -> Image.Image:
    scale = min(canvas_width / image.width, canvas_height / image.height)
    display_size = (
        max(1, min(canvas_width, round(image.width * scale))),
        max(1, min(canvas_height, round(image.height * scale))),
    )
    rendered = image.resize(display_size, Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    canvas.paste(rendered, ((canvas_width - display_size[0]) // 2, (canvas_height - display_size[1]) // 2))
    encoded = BytesIO()
    canvas.save(encoded, format="PNG")
    encoded.seek(0)
    with Image.open(encoded) as captured:
        captured.load()
        return captured.convert("RGB").resize(image.size, Image.Resampling.BICUBIC)
