"""Conservative single-image heuristics for visible processing signals.

These checks do not recover edit history. They surface pixel-level clues that are useful in the
demo while explicitly leaving ambiguous transformations unresolved.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from PIL import Image, ImageFilter

from prooflens.errors import UserInputError


@dataclass(frozen=True, slots=True)
class ProcessingAssessment:
    detected: bool
    likely_transformations: tuple[str, ...]
    confidence: str
    evidence: tuple[str, ...]
    caveat: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def assess_processing(image: Image.Image) -> ProcessingAssessment:
    """Estimate visible processing clues without claiming unavailable edit provenance."""

    if not isinstance(image, Image.Image):
        raise UserInputError("processing assessment requires a PIL image")
    rgb = image.convert("RGB")
    width, height = rgb.size
    if width < 2 or height < 2:
        raise UserInputError("processing assessment requires an image of at least 2 by 2 pixels")

    grayscale = np.asarray(rgb.convert("L"), dtype=np.float32) / 255.0
    labels: list[str] = []
    evidence: list[str] = []
    strong_signal = False

    image_format = str(image.format or "").upper()
    blockiness = _jpeg_blockiness(grayscale)
    if image_format in {"JPEG", "JPG"} or blockiness >= 1.35:
        labels.append("JPEG encoding or compression")
        if image_format in {"JPEG", "JPG"}:
            evidence.append("The uploaded file is JPEG encoded.")
        if blockiness >= 1.35:
            evidence.append("An 8-pixel block-boundary pattern is visible.")
            strong_signal = True

    sharpness = _laplacian_variance(grayscale)
    if min(width, height) >= 64 and sharpness < 0.0012:
        labels.append("Strong blur or loss of detail")
        evidence.append("Local edge variation is unusually low.")
        strong_signal = True

    noise_score = _noise_residual(grayscale)
    if min(width, height) >= 64 and noise_score > 0.075 and sharpness > 0.025:
        labels.append("Noise-like high-frequency pattern")
        evidence.append("High-frequency residual energy is elevated.")

    if max(width, height) < 384:
        labels.append("Low resolution; prior rescaling is possible")
        evidence.append(f"Image dimensions are {width} × {height} pixels.")

    if not labels:
        return ProcessingAssessment(
            detected=False,
            likely_transformations=(),
            confidence="low",
            evidence=("No strong compression, blur, noise, or low-resolution signal was found.",),
            caveat=_CAVEAT,
        )
    return ProcessingAssessment(
        detected=True,
        likely_transformations=tuple(labels),
        confidence="moderate" if strong_signal else "low",
        evidence=tuple(evidence),
        caveat=_CAVEAT,
    )


def _jpeg_blockiness(grayscale: np.ndarray) -> float:
    if min(grayscale.shape) < 24:
        return 0.0
    horizontal = np.abs(np.diff(grayscale, axis=1))
    vertical = np.abs(np.diff(grayscale, axis=0))
    boundary_x = np.arange(7, horizontal.shape[1], 8)
    boundary_y = np.arange(7, vertical.shape[0], 8)
    if not len(boundary_x) or not len(boundary_y):
        return 0.0
    boundary = np.concatenate(
        (horizontal[:, boundary_x].ravel(), vertical[boundary_y, :].ravel())
    )
    interior_x = np.setdiff1d(np.arange(horizontal.shape[1]), boundary_x)
    interior_y = np.setdiff1d(np.arange(vertical.shape[0]), boundary_y)
    interior = np.concatenate(
        (horizontal[:, interior_x].ravel(), vertical[interior_y, :].ravel())
    )
    return float((boundary.mean() + 1e-6) / (interior.mean() + 1e-6))


def _laplacian_variance(grayscale: np.ndarray) -> float:
    center = grayscale[1:-1, 1:-1]
    laplacian = (
        grayscale[:-2, 1:-1]
        + grayscale[2:, 1:-1]
        + grayscale[1:-1, :-2]
        + grayscale[1:-1, 2:]
        - 4.0 * center
    )
    return float(laplacian.var()) if laplacian.size else 0.0


def _noise_residual(grayscale: np.ndarray) -> float:
    source = Image.fromarray(np.uint8(np.clip(grayscale * 255.0, 0, 255)), mode="L")
    smoothed = np.asarray(source.filter(ImageFilter.GaussianBlur(radius=1.0)), dtype=np.float32)
    return float(np.std(grayscale - smoothed / 255.0))


_CAVEAT = (
    "This is a heuristic estimate from one final image, not edit-history proof. Cropping, color "
    "adjustment, and rescaling may be indistinguishable without the original image."
)
