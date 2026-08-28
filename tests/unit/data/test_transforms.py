from __future__ import annotations

import builtins
import socket
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest
from PIL import Image

import prooflens.data.transforms as transforms_module
from prooflens.data.transforms import (
    TransformSpec,
    apply_transform,
    canonical_specs,
    gaussian_blur_kernel_size,
    get_spec,
    group_specs_by_family,
    sample_color_jitter_factors,
    training_condition_probabilities,
)
from prooflens.errors import DataIntegrityError, UserInputError

EXPECTED_SPECS = (
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
EXPECTED_FAMILIES = (
    "jpeg",
    "blur",
    "resize",
    "noise",
    "color_jitter",
    "center_crop",
)


def _detail_image(width: int = 96, height: int = 80) -> Image.Image:
    y, x = np.indices((height, width))
    checker = ((x // 3 + y // 3) % 2) * 100
    pixels = np.stack(
        (
            (x * 11 + y * 3 + checker) % 256,
            (x * 5 + y * 13 + checker // 2) % 256,
            (x * 17 + y * 7 + checker) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _edge_energy(pixels: np.ndarray) -> float:
    values = pixels.astype(np.float64)
    horizontal = np.abs(np.diff(values, axis=1)).mean()
    vertical = np.abs(np.diff(values, axis=0)).mean()
    return float(horizontal + vertical)


def _mse(first: np.ndarray, second: np.ndarray) -> float:
    difference = first.astype(np.float64) - second.astype(np.float64)
    return float(np.mean(difference**2))


def test_canonical_registry_has_exact_stable_specs_in_order() -> None:
    observed = tuple(
        (spec.condition_id, spec.family, spec.severity, dict(spec.parameters))
        for spec in canonical_specs()
    )

    assert observed == EXPECTED_SPECS
    assert len({spec.condition_id for spec in canonical_specs()}) == 14


def test_specs_and_parameter_mappings_are_not_externally_mutable() -> None:
    source_parameters = {"sigma": 0.5}
    spec = TransformSpec("blur", "blur_s0.5", 0.5, source_parameters)
    source_parameters["sigma"] = 2.0

    assert spec.parameters["sigma"] == 0.5
    with pytest.raises(TypeError):
        spec.parameters["sigma"] = 2.0  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        spec.severity = 2.0  # type: ignore[misc]


def test_get_spec_returns_canonical_spec_and_rejects_unknown_ids() -> None:
    assert get_spec("noise_s0.05") is canonical_specs()[10]

    with pytest.raises(UserInputError, match=r"unknown transform condition.*missing_condition"):
        get_spec("missing_condition")


def test_group_specs_by_family_is_deterministic() -> None:
    grouped = group_specs_by_family()

    assert tuple(grouped) == EXPECTED_FAMILIES
    assert tuple(
        spec.condition_id for family in grouped.values() for spec in family
    ) == tuple(item[0] for item in EXPECTED_SPECS)
    assert all(isinstance(specs, tuple) for specs in grouped.values())


def test_training_probabilities_are_uniform_within_six_equal_mass_families() -> None:
    probabilities = training_condition_probabilities()
    grouped = group_specs_by_family()

    assert tuple(probabilities) == tuple(item[0] for item in EXPECTED_SPECS)
    assert sum(probabilities.values()) == 1.0
    for specs in grouped.values():
        values = [probabilities[spec.condition_id] for spec in specs]
        assert values == pytest.approx([1 / (6 * len(specs))] * len(specs))
        assert sum(values) == pytest.approx(1 / 6)


def test_training_probabilities_reject_a_missing_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    without_noise = tuple(
        spec for spec in canonical_specs() if spec.family != "noise"
    )
    monkeypatch.setattr(transforms_module, "_CANONICAL_SPECS", without_noise)

    with pytest.raises(DataIntegrityError, match=r"exactly six.*missing.*noise"):
        training_condition_probabilities()


def test_training_probabilities_reject_duplicate_condition_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duplicate = canonical_specs() + (canonical_specs()[0],)
    monkeypatch.setattr(transforms_module, "_CANONICAL_SPECS", duplicate)

    with pytest.raises(DataIntegrityError, match=r"condition IDs.*unique.*jpeg_q90"):
        training_condition_probabilities()


def test_every_transform_returns_new_rgb_image_at_original_size_without_mutation() -> None:
    image = _detail_image(79, 61)
    original = np.asarray(image).copy()

    for spec in canonical_specs():
        transformed = apply_transform(image, spec, seed=17)

        assert transformed is not image
        assert transformed.mode == "RGB"
        assert transformed.size == image.size
        assert np.array_equal(np.asarray(image), original)


def test_every_transform_is_repeatable_for_identical_image_spec_and_seed() -> None:
    image = _detail_image()

    for spec in canonical_specs():
        first = np.asarray(apply_transform(image, spec, seed=901))
        second = np.asarray(apply_transform(image, spec, seed=901))
        assert np.array_equal(first, second), spec.condition_id


@pytest.mark.parametrize("condition_id", ["noise_s0.05", "color_jitter_20"])
def test_stochastic_transforms_change_with_seed(condition_id: str) -> None:
    image = _detail_image()
    spec = get_spec(condition_id)

    first = np.asarray(apply_transform(image, spec, seed=11))
    second = np.asarray(apply_transform(image, spec, seed=12))

    assert not np.array_equal(first, second)


def test_lower_jpeg_quality_increases_reconstruction_error() -> None:
    image = _detail_image()
    original = np.asarray(image)
    high_quality = np.asarray(apply_transform(image, get_spec("jpeg_q90"), seed=1))
    low_quality = np.asarray(apply_transform(image, get_spec("jpeg_q30"), seed=1))

    assert _mse(original, low_quality) > _mse(original, high_quality)


def test_stronger_blur_reduces_edge_energy() -> None:
    image = _detail_image()
    mild = np.asarray(apply_transform(image, get_spec("blur_s0.5"), seed=1))
    strong = np.asarray(apply_transform(image, get_spec("blur_s2.0"), seed=1))

    assert _edge_energy(strong) < _edge_energy(mild)


def test_more_aggressive_resize_reduces_detail() -> None:
    image = _detail_image()
    half = np.asarray(apply_transform(image, get_spec("resize_x0.5"), seed=1))
    quarter = np.asarray(apply_transform(image, get_spec("resize_x0.25"), seed=1))

    assert _edge_energy(quarter) < _edge_energy(half)


def test_stronger_noise_increases_flat_field_variance() -> None:
    image = Image.new("RGB", (128, 96), (128, 128, 128))
    mild = np.asarray(apply_transform(image, get_spec("noise_s0.02"), seed=17))
    strong = np.asarray(apply_transform(image, get_spec("noise_s0.10"), seed=17))

    assert float(strong.astype(np.float64).var()) > float(
        mild.astype(np.float64).var()
    )


def test_center_crop_retains_exact_central_eighty_percent_then_restores_size() -> None:
    y, x = np.indices((10, 10))
    pixels = np.stack((x * 20, y * 20, np.full_like(x, 100)), axis=-1).astype(
        np.uint8
    )
    image = Image.fromarray(pixels, mode="RGB")
    expected = image.crop((1, 1, 9, 9)).resize((10, 10), Image.Resampling.BICUBIC)

    transformed = apply_transform(image, get_spec("center_crop_80"), seed=1)

    assert np.array_equal(np.asarray(transformed), np.asarray(expected))


def test_color_jitter_factors_are_independent_repeatable_and_within_twenty_percent() -> None:
    sampled = [sample_color_jitter_factors(seed) for seed in range(32)]

    assert sample_color_jitter_factors(7) == sample_color_jitter_factors(7)
    assert sample_color_jitter_factors(7) != sample_color_jitter_factors(8)
    assert all(0.8 <= factor <= 1.2 for factors in sampled for factor in factors)
    assert any(len(set(factors)) == 3 for factors in sampled)


@pytest.mark.parametrize(
    ("sigma", "expected"),
    [(0.5, 5), (1.0, 7), (2.0, 13)],
)
def test_gaussian_blur_kernel_covers_three_sigma_with_odd_size(
    sigma: float, expected: int
) -> None:
    kernel_size = gaussian_blur_kernel_size(sigma)

    assert kernel_size == expected
    assert kernel_size % 2 == 1


@pytest.mark.parametrize(
    "image",
    [
        Image.new("L", (1, 1), 127),
        Image.new("L", (1, 7), 127),
        Image.new("RGBA", (7, 1), (20, 40, 60, 80)),
    ],
)
def test_all_transforms_handle_grayscale_rgba_and_tiny_dimensions(
    image: Image.Image,
) -> None:
    for spec in canonical_specs():
        transformed = apply_transform(image, spec, seed=3)
        assert transformed.mode == "RGB"
        assert transformed.size == image.size


@pytest.mark.parametrize("image", [None, np.zeros((4, 4, 3)), "image.png"])
def test_apply_transform_rejects_malformed_image_inputs(image: Any) -> None:
    with pytest.raises(UserInputError, match="PIL image"):
        apply_transform(image, get_spec("jpeg_q90"), seed=1)


@pytest.mark.parametrize("seed", [-1, 1.5, True])
def test_apply_transform_rejects_invalid_seeds(seed: Any) -> None:
    with pytest.raises(UserInputError, match="nonnegative integer seed"):
        apply_transform(_detail_image(), get_spec("noise_s0.02"), seed=seed)


@pytest.mark.parametrize(
    ("family", "condition_id", "severity", "parameters"),
    [
        ("unsupported", "unsupported_1", 1.0, {"value": 1.0}),
        ("jpeg", "jpeg_q101", 101.0, {"quality": 101, "subsampling": 2}),
        ("jpeg", "jpeg_q90", 90.0, {"quality": 90, "subsampling": 1}),
        ("blur", "blur_s0", 0.0, {"sigma": 0.0}),
        ("resize", "resize_x1.0", 1.0, {"scale": 1.0}),
        ("noise", "noise_s0", 0.0, {"sigma": 0.0}),
        ("color_jitter", "color_jitter_30", 0.3, {"magnitude": 0.3}),
        ("center_crop", "center_crop_50", 0.5, {"fraction": 0.5}),
        ("blur", "blur_s1.0", 1.0, {"sigma": 2.0}),
    ],
)
def test_transform_spec_rejects_unsupported_families_severities_and_parameters(
    family: Any,
    condition_id: str,
    severity: float,
    parameters: dict[str, float | int | str],
) -> None:
    with pytest.raises(UserInputError, match="transform spec"):
        TransformSpec(family, condition_id, severity, parameters)


def test_transforms_do_not_use_filesystem_or_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("external I/O is forbidden")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    image = _detail_image(24, 20)

    for spec in canonical_specs():
        transformed = apply_transform(image, spec, seed=19)
        assert transformed.size == image.size
