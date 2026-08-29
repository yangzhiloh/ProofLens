from __future__ import annotations

import hashlib
import math
import random
import struct
from numbers import Integral
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from prooflens.data.transforms import (
    TransformSpec,
    get_spec,
    group_specs_by_family,
)
from prooflens.errors import DataIntegrityError

_SEED_DOMAIN = b"prooflens-stable-seed-v1\0"
_MAX_SEED = 2**63


def compute_sampling_weights(frame: pd.DataFrame) -> np.ndarray:
    """Return positional weights with equal label and within-label stratum mass."""

    if not isinstance(frame, pd.DataFrame):
        raise DataIntegrityError("sampling weights require a pandas DataFrame")
    required = {"label", "dataset_name", "generator_family"}
    missing = required - set(frame.columns)
    if missing:
        raise DataIntegrityError(
            f"sampling weights are missing required fields: {sorted(missing)}"
        )

    labels: list[int] = []
    strata: list[tuple[object, ...]] = []
    for position in range(len(frame)):
        row = frame.iloc[position]
        label = _binary_label(row["label"], position)
        dataset_name = _stratum_text(row["dataset_name"], "dataset_name", position)
        if label == 0:
            stratum = (label, dataset_name)
        else:
            generator = _stratum_text(
                row["generator_family"], "generator_family", position
            )
            stratum = (label, dataset_name, generator)
        labels.append(label)
        strata.append(stratum)
    if set(labels) != {0, 1}:
        raise DataIntegrityError("sampling weights require both labels 0 and 1")

    weights = np.zeros(len(frame), dtype=np.float64)
    for label in (0, 1):
        names = sorted({strata[index] for index, value in enumerate(labels) if value == label})
        for name in names:
            positions = [index for index, value in enumerate(strata) if value == name]
            row_weight = 0.5 / (len(names) * len(positions))
            weights[positions] = row_weight
    if (
        not np.isfinite(weights).all()
        or not (weights > 0).all()
        or not math.isclose(float(weights.sum()), 1.0, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise DataIntegrityError("sampling weights must be finite, positive, and total 1")
    return weights


def make_weighted_sampler(
    frame: pd.DataFrame,
    *,
    seed: int,
    num_samples: int | None = None,
    replacement: bool = True,
) -> WeightedRandomSampler:
    """Construct a sampler whose private generator never advances global RNG state."""

    normalized_seed = _nonnegative_seed(seed, "sampler seed")
    if num_samples is None:
        num_samples = len(frame)
    if (
        not isinstance(num_samples, Integral)
        or isinstance(num_samples, bool)
        or int(num_samples) < 1
    ):
        raise DataIntegrityError("num_samples must be a positive integer")
    if not isinstance(replacement, bool):
        raise DataIntegrityError("replacement must be a boolean")
    generator = torch.Generator()
    generator.manual_seed(normalized_seed)
    weights = torch.as_tensor(compute_sampling_weights(frame), dtype=torch.float64)
    return WeightedRandomSampler(
        weights,
        num_samples=int(num_samples),
        replacement=replacement,
        generator=generator,
    )


def stable_seed(*components: object) -> int:
    """Hash typed primitive components into a stable nonnegative 63-bit seed."""

    payload = bytearray(_SEED_DOMAIN)
    payload.extend(len(components).to_bytes(8, "big"))
    for component in components:
        tag, encoded = _encode_component(component)
        payload.extend(tag)
        payload.extend(len(encoded).to_bytes(8, "big"))
        payload.extend(encoded)
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") & (_MAX_SEED - 1)


@runtime_checkable
class TransformSampler(Protocol):
    def sample(self, sample_id: str, *, epoch: int, seed: int) -> TransformSpec:
        """Select one canonical transform deterministically for an item."""


class FixedTransformSampler:
    """Return one canonical condition for every item, useful in tests and evaluation."""

    def __init__(self, condition_id: str) -> None:
        self._spec = get_spec(condition_id)

    def sample(self, sample_id: str, *, epoch: int, seed: int) -> TransformSpec:
        _sampling_identity(sample_id, epoch, seed)
        return self._spec


class FamilyBalancedTransformSampler:
    """Select a family uniformly, then a severity uniformly within that family."""

    def __init__(self) -> None:
        grouped = group_specs_by_family()
        self._families: tuple[tuple[TransformSpec, ...], ...] = tuple(grouped.values())

    def sample(self, sample_id: str, *, epoch: int, seed: int) -> TransformSpec:
        normalized_epoch, normalized_seed = _sampling_identity(sample_id, epoch, seed)
        rng = random.Random(
            stable_seed(normalized_seed, normalized_epoch, sample_id, "transform-spec")
        )
        family = self._families[rng.randrange(len(self._families))]
        return family[rng.randrange(len(family))]


def _sampling_identity(sample_id: object, epoch: object, seed: object) -> tuple[int, int]:
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise DataIntegrityError("transform sample_id must be a nonempty string")
    normalized_epoch = _nonnegative_seed(epoch, "transform epoch")
    normalized_seed = _nonnegative_seed(seed, "transform seed")
    return normalized_epoch, normalized_seed


def _encode_component(component: object) -> tuple[bytes, bytes]:
    if component is None:
        return b"n", b""
    if isinstance(component, bool):
        return b"b", b"\x01" if component else b"\x00"
    if isinstance(component, int):
        return b"i", str(component).encode("ascii")
    if isinstance(component, float):
        if not math.isfinite(component):
            raise DataIntegrityError("stable seed component floats must be finite")
        return b"f", struct.pack(">d", component)
    if isinstance(component, str):
        return b"s", component.encode("utf-8")
    if isinstance(component, bytes):
        return b"y", component
    raise DataIntegrityError(
        "stable seed component values must be None, bool, int, finite float, str, or bytes"
    )


def _binary_label(value: object, position: int) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) not in (0, 1):
        raise DataIntegrityError(
            f"label at row position {position} must be binary 0 or 1"
        )
    return int(value)


def _stratum_text(value: object, field: str, position: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataIntegrityError(
            f"{field} at row position {position} must be nonblank stratum metadata"
        )
    return value


def _nonnegative_seed(value: object, field: str) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, bool)
        or not 0 <= int(value) < _MAX_SEED
    ):
        raise DataIntegrityError(f"{field} must be a nonnegative integer below 2**63")
    return int(value)
