from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from io import BytesIO
from numbers import Integral, Real
from types import MappingProxyType
from typing import Literal

import numpy as np
from PIL import Image, ImageEnhance

from prooflens.errors import DataIntegrityError, UserInputError

TransformFamily = Literal[
    "jpeg",
    "blur",
    "resize",
    "noise",
    "color_jitter",
    "center_crop",
]
ParameterValue = float | int | str

_FAMILY_ORDER: tuple[TransformFamily, ...] = (
    "jpeg",
    "blur",
    "resize",
    "noise",
    "color_jitter",
    "center_crop",
)
_SPEC_DEFINITIONS: tuple[
    tuple[str, TransformFamily, float, dict[str, ParameterValue]], ...
] = (
    ("jpeg_q90", "jpeg", 90.0, {"quality": 90, "subsampling": 2}),
    ("jpeg_q70", "jpeg", 70.0, {"quality": 70, "subsampling": 2}),
    ("jpeg_q50", "jpeg", 50.0, {"quality": 50, "subsampling": 2}),
    ("jpeg_q30", "jpeg", 30.0, {"quality": 30, "subsampling": 2}),
    ("blur_s0.5", "blur", 0.5, {"sigma": 0.5}),
    ("blur_s1.0", "blur", 1.0, {"sigma": 1.0}),
    ("blur_s2.0", "blur", 2.0, {"sigma": 2.0}),
    ("resize_x0.5", "resize", 0.5, {"scale": 0.5}),
    ("resize_x0.25", "resize", 0.25, {"scale": 0.25}),
    ("noise_s0.02", "noise", 0.02, {"sigma": 0.02}),
    ("noise_s0.05", "noise", 0.05, {"sigma": 0.05}),
    ("noise_s0.10", "noise", 0.10, {"sigma": 0.10}),
    ("color_jitter_20", "color_jitter", 0.20, {"magnitude": 0.20}),
    ("center_crop_80", "center_crop", 0.80, {"fraction": 0.80}),
)
_EXPECTED_SPECS = {
    condition_id: (family, severity, parameters)
    for condition_id, family, severity, parameters in _SPEC_DEFINITIONS
}


@dataclass(frozen=True, slots=True)
class TransformSpec:
    """One immutable canonical robustness condition."""

    family: TransformFamily
    condition_id: str
    severity: float
    parameters: Mapping[str, ParameterValue]

    def __post_init__(self) -> None:
        parameters = _validated_parameter_copy(self.parameters)
        _validate_spec_fields(
            self.family,
            self.condition_id,
            self.severity,
            parameters,
        )
        object.__setattr__(self, "severity", float(self.severity))
        object.__setattr__(self, "parameters", MappingProxyType(parameters))


def canonical_specs() -> tuple[TransformSpec, ...]:
    """Return the stable, immutable canonical transform registry."""

    _validate_registry(_CANONICAL_SPECS)
    return _CANONICAL_SPECS


def get_spec(condition_id: str) -> TransformSpec:
    """Resolve a stable condition ID or raise a readable typed input error."""

    if not isinstance(condition_id, str) or not condition_id.strip():
        raise UserInputError("transform condition ID must be a nonempty string")
    for spec in canonical_specs():
        if spec.condition_id == condition_id:
            return spec
    raise UserInputError(f"unknown transform condition {condition_id!r}")


def group_specs_by_family() -> dict[TransformFamily, tuple[TransformSpec, ...]]:
    """Group canonical specs in stable family and registry order."""

    specs = canonical_specs()
    return {
        family: tuple(spec for spec in specs if spec.family == family)
        for family in _FAMILY_ORDER
    }


def training_condition_probabilities() -> dict[str, float]:
    """Weight six families equally and severities uniformly within each family."""

    grouped = group_specs_by_family()
    family_count = len(_FAMILY_ORDER)
    probabilities = {
        spec.condition_id: 1.0 / (family_count * len(grouped[spec.family]))
        for spec in canonical_specs()
    }
    _validate_probability_contract(probabilities, grouped)
    return probabilities


def gaussian_blur_kernel_size(sigma: float) -> int:
    """Return the odd kernel width spanning three sigma on either side."""

    if (
        not isinstance(sigma, Real)
        or isinstance(sigma, bool)
        or not math.isfinite(float(sigma))
        or float(sigma) <= 0
    ):
        raise UserInputError("Gaussian blur sigma must be a finite positive number")
    return 2 * math.ceil(3 * float(sigma)) + 1


def sample_color_jitter_factors(
    seed: int, magnitude: float = 0.20
) -> tuple[float, float, float]:
    """Sample independent brightness, contrast, and saturation factors."""

    normalized_seed = _validated_seed(seed)
    if (
        not isinstance(magnitude, Real)
        or isinstance(magnitude, bool)
        or not math.isfinite(float(magnitude))
        or not 0 <= float(magnitude) <= 1
    ):
        raise UserInputError("color jitter magnitude must be a finite number from 0 to 1")
    factors = np.random.default_rng(normalized_seed).uniform(
        1.0 - float(magnitude),
        1.0 + float(magnitude),
        size=3,
    )
    return tuple(float(factor) for factor in factors)  # type: ignore[return-value]


def apply_transform(image: Image.Image, spec: TransformSpec, seed: int) -> Image.Image:
    """Apply one canonical transform without mutating the source image."""

    rgb = _validated_rgb_copy(image)
    normalized_seed = _validated_seed(seed)
    _validate_spec_instance(spec)

    if spec.family == "jpeg":
        transformed = _apply_jpeg(rgb, spec)
    elif spec.family == "blur":
        transformed = _apply_blur(rgb, spec)
    elif spec.family == "resize":
        transformed = _apply_resize(rgb, spec)
    elif spec.family == "noise":
        transformed = _apply_noise(rgb, spec, normalized_seed)
    elif spec.family == "color_jitter":
        transformed = _apply_color_jitter(rgb, spec, normalized_seed)
    elif spec.family == "center_crop":
        transformed = _apply_center_crop(rgb, spec)
    else:  # pragma: no cover - guarded by TransformSpec validation
        raise UserInputError(f"unsupported transform family {spec.family!r}")

    if transformed.mode != "RGB" or transformed.size != rgb.size:
        raise DataIntegrityError(
            f"transform {spec.condition_id!r} violated the RGB dimension contract"
        )
    return transformed


def _apply_jpeg(image: Image.Image, spec: TransformSpec) -> Image.Image:
    buffer = BytesIO()
    image.save(
        buffer,
        format="JPEG",
        quality=int(spec.parameters["quality"]),
        subsampling=int(spec.parameters["subsampling"]),
    )
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        decoded.load()
        return decoded.convert("RGB").copy()


def _apply_blur(image: Image.Image, spec: TransformSpec) -> Image.Image:
    sigma = float(spec.parameters["sigma"])
    kernel_size = gaussian_blur_kernel_size(sigma)
    radius = kernel_size // 2
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(coordinates**2) / (2.0 * sigma**2))
    kernel /= kernel.sum()

    pixels = np.asarray(image, dtype=np.float64)
    horizontal_padding = np.pad(pixels, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    horizontal = sum(
        kernel[offset] * horizontal_padding[:, offset : offset + image.width, :]
        for offset in range(kernel_size)
    )
    vertical_padding = np.pad(horizontal, ((radius, radius), (0, 0), (0, 0)), mode="edge")
    blurred = sum(
        kernel[offset] * vertical_padding[offset : offset + image.height, :, :]
        for offset in range(kernel_size)
    )
    return Image.fromarray(np.clip(np.rint(blurred), 0, 255).astype(np.uint8), mode="RGB")


def _apply_resize(image: Image.Image, spec: TransformSpec) -> Image.Image:
    scale = float(spec.parameters["scale"])
    downsampled_size = (
        max(1, round(image.width * scale)),
        max(1, round(image.height * scale)),
    )
    downsampled = image.resize(downsampled_size, Image.Resampling.BICUBIC)
    return downsampled.resize(image.size, Image.Resampling.BICUBIC)


def _apply_noise(image: Image.Image, spec: TransformSpec, seed: int) -> Image.Image:
    sigma = float(spec.parameters["sigma"])
    pixels = np.asarray(image, dtype=np.float64) / 255.0
    noise = np.random.default_rng(seed).normal(0.0, sigma, size=pixels.shape)
    noisy = np.clip(pixels + noise, 0.0, 1.0)
    return Image.fromarray(np.rint(noisy * 255.0).astype(np.uint8), mode="RGB")


def _apply_color_jitter(
    image: Image.Image, spec: TransformSpec, seed: int
) -> Image.Image:
    magnitude = float(spec.parameters["magnitude"])
    brightness, contrast, saturation = sample_color_jitter_factors(seed, magnitude)
    transformed = ImageEnhance.Brightness(image).enhance(brightness)
    transformed = ImageEnhance.Contrast(transformed).enhance(contrast)
    return ImageEnhance.Color(transformed).enhance(saturation).convert("RGB")


def _apply_center_crop(image: Image.Image, spec: TransformSpec) -> Image.Image:
    fraction = float(spec.parameters["fraction"])
    crop_width = max(1, round(image.width * fraction))
    crop_height = max(1, round(image.height * fraction))
    left = (image.width - crop_width) // 2
    top = (image.height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize(image.size, Image.Resampling.BICUBIC)


def _validated_parameter_copy(
    parameters: Mapping[str, ParameterValue],
) -> dict[str, ParameterValue]:
    if not isinstance(parameters, Mapping):
        raise UserInputError("invalid transform spec: parameters must be a mapping")
    copied = dict(parameters)
    if any(not isinstance(key, str) or not key for key in copied):
        raise UserInputError("invalid transform spec: parameter names must be nonempty strings")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float, str))
        for value in copied.values()
    ):
        raise UserInputError("invalid transform spec: parameter values must be scalar")
    return copied


def _validate_spec_fields(
    family: object,
    condition_id: object,
    severity: object,
    parameters: dict[str, ParameterValue],
) -> None:
    if family not in _FAMILY_ORDER:
        raise UserInputError(f"invalid transform spec: unsupported family {family!r}")
    if not isinstance(condition_id, str) or not condition_id:
        raise UserInputError("invalid transform spec: condition ID must be a nonempty string")
    if (
        not isinstance(severity, Real)
        or isinstance(severity, bool)
        or not math.isfinite(float(severity))
    ):
        raise UserInputError("invalid transform spec: severity must be a finite number")
    expected = _EXPECTED_SPECS.get(condition_id)
    if expected is None:
        raise UserInputError(f"invalid transform spec: unknown condition ID {condition_id!r}")
    expected_family, expected_severity, expected_parameters = expected
    if (
        family != expected_family
        or float(severity) != expected_severity
        or parameters != expected_parameters
    ):
        raise UserInputError(
            f"invalid transform spec: {condition_id!r} does not match its canonical definition"
        )


def _validate_spec_instance(spec: object) -> None:
    if not isinstance(spec, TransformSpec):
        raise UserInputError("apply_transform requires a TransformSpec")
    parameters = _validated_parameter_copy(spec.parameters)
    _validate_spec_fields(spec.family, spec.condition_id, spec.severity, parameters)


def _validate_registry(specs: tuple[TransformSpec, ...]) -> None:
    if not isinstance(specs, tuple) or any(
        not isinstance(spec, TransformSpec) for spec in specs
    ):
        raise DataIntegrityError("canonical transform registry must be a tuple of TransformSpec")
    identifiers = [spec.condition_id for spec in specs]
    duplicates = sorted(
        condition_id
        for condition_id, count in Counter(identifiers).items()
        if count > 1
    )
    if duplicates:
        raise DataIntegrityError(
            f"canonical transform condition IDs must be unique; duplicates: {duplicates}"
        )
    observed_families = {spec.family for spec in specs}
    required_families = set(_FAMILY_ORDER)
    if observed_families != required_families:
        missing = sorted(required_families - observed_families)
        extra = sorted(observed_families - required_families)
        raise DataIntegrityError(
            "canonical transform registry must contain exactly six families; "
            f"missing: {missing}; extra: {extra}"
        )
    expected_ids = [definition[0] for definition in _SPEC_DEFINITIONS]
    if identifiers != expected_ids:
        raise DataIntegrityError(
            "canonical transform registry condition IDs or ordering differ from the contract"
        )


def _validate_probability_contract(
    probabilities: dict[str, float],
    grouped: dict[TransformFamily, tuple[TransformSpec, ...]],
) -> None:
    if tuple(grouped) != _FAMILY_ORDER or any(not specs for specs in grouped.values()):
        raise DataIntegrityError(
            "training probabilities require each canonical family exactly once and nonempty"
        )
    if not math.isclose(sum(probabilities.values()), 1.0, rel_tol=0.0, abs_tol=1e-15):
        raise DataIntegrityError("training condition probabilities must total 1.0")
    for family, specs in grouped.items():
        family_mass = sum(probabilities[spec.condition_id] for spec in specs)
        if not math.isclose(family_mass, 1.0 / 6.0, rel_tol=0.0, abs_tol=1e-15):
            raise DataIntegrityError(
                f"training condition probability mass for {family!r} must equal 1/6"
            )


def _validated_rgb_copy(image: object) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise UserInputError("apply_transform requires a PIL image")
    try:
        if image.width < 1 or image.height < 1:
            raise ValueError("image has empty dimensions")
        converted = image.convert("RGB")
        converted.load()
        return converted.copy()
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise UserInputError("apply_transform requires a decodable PIL image") from error


def _validated_seed(seed: object) -> int:
    if not isinstance(seed, Integral) or isinstance(seed, bool) or int(seed) < 0:
        raise UserInputError("transform seed must be a nonnegative integer seed")
    return int(seed)


_CANONICAL_SPECS = tuple(
    TransformSpec(family, condition_id, severity, parameters)
    for condition_id, family, severity, parameters in _SPEC_DEFINITIONS
)
