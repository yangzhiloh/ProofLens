"""Deterministic, loss-guided transformation candidate selection."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import torch
from PIL import Image
from torch import Tensor

from prooflens.data.sampling import stable_seed
from prooflens.data.transforms import TransformSpec, canonical_specs, group_specs_by_family
from prooflens.errors import DataIntegrityError


@dataclass(frozen=True, slots=True)
class HardTransformSelection:
    condition_ids: tuple[str, ...]
    transformed_images: tuple[Image.Image, ...]


def select_lowest_margin(logits_by_condition: dict[str, Tensor], labels: Tensor) -> tuple[str, ...]:
    if not logits_by_condition:
        raise ValueError("at least one candidate condition is required")
    names = tuple(logits_by_condition)
    logits = torch.stack([logits_by_condition[name] for name in names], dim=1)
    if logits.ndim != 2 or logits.shape[0] != labels.numel():
        raise ValueError("candidate logits must have shape [batch, candidate_count]")
    indices = torch.argmin(correct_margin_matrix(logits, labels), dim=1).tolist()
    return tuple(names[index] for index in indices)


class HardTransformMiner:
    def __init__(
        self,
        registry: Sequence[TransformSpec] | None = None,
        seed: int = 17,
        candidate_count: int = 3,
        exploration_probability: float = 0.20,
    ) -> None:
        specs = tuple(registry) if registry is not None else canonical_specs()
        grouped = {
            family: tuple(values)
            for family, values in _group_registry(specs).items()
        }
        if not 1 <= candidate_count <= len(grouped):
            raise ValueError("candidate_count must be between 1 and the number of families")
        if not 0 <= exploration_probability <= 1:
            raise ValueError("exploration_probability must be between 0 and 1")
        self.by_family = grouped
        self.seed = int(seed)
        self.candidate_count = int(candidate_count)
        self.exploration_probability = float(exploration_probability)

    def sample_candidates(
        self, sample_ids: Sequence[str], epoch: int
    ) -> tuple[tuple[TransformSpec, ...], ...]:
        families = tuple(self.by_family)
        selections = []
        for sample_id in sample_ids:
            rng = random.Random(stable_seed(self.seed, epoch, sample_id))
            selected = rng.sample(families, k=self.candidate_count)
            selections.append(tuple(rng.choice(self.by_family[name]) for name in selected))
        return tuple(selections)

    def select(
        self,
        candidate_logits: Tensor,
        candidate_condition_ids: Sequence[Sequence[str]],
        labels: Tensor,
        sample_ids: Sequence[str],
        epoch: int,
    ) -> tuple[str, ...]:
        if candidate_logits.shape != (len(labels), self.candidate_count):
            raise ValueError("candidate logits must have shape [batch, candidate_count]")
        if len(candidate_condition_ids) != len(labels) or len(sample_ids) != len(labels):
            raise ValueError("candidate metadata must match the batch size")
        if any(len(row) != self.candidate_count for row in candidate_condition_ids):
            raise ValueError("each candidate metadata row must match candidate_count")
        margins = correct_margin_matrix(candidate_logits, labels)
        indices = margins.argmin(dim=1).tolist()
        for row, sample_id in enumerate(sample_ids):
            rng = random.Random(stable_seed(self.seed, epoch, sample_id, "explore"))
            if rng.random() < self.exploration_probability:
                indices[row] = rng.randrange(self.candidate_count)
        return tuple(
            str(candidate_condition_ids[row][column]) for row, column in enumerate(indices)
        )


def correct_margin_matrix(candidate_logits: Tensor, labels: Tensor) -> Tensor:
    if candidate_logits.ndim != 2 or labels.ndim != 1:
        raise ValueError("candidate logits must be rank 2 and labels rank 1")
    if candidate_logits.shape[0] != labels.shape[0]:
        raise ValueError("candidate logits and labels must have matching batch sizes")
    return torch.where(labels[:, None] == 1, candidate_logits, -candidate_logits)


def _group_registry(specs: Sequence[TransformSpec]) -> dict[str, tuple[TransformSpec, ...]]:
    if not specs:
        raise DataIntegrityError("hard-transform registry cannot be empty")
    # group_specs_by_family is the canonical registry implementation; retaining
    # this local path permits focused tests to provide a small registry.
    canonical = tuple(canonical_specs())
    if tuple(specs) == canonical:
        return group_specs_by_family()
    grouped: dict[str, list[TransformSpec]] = {}
    for spec in specs:
        grouped.setdefault(spec.family, []).append(spec)
    return {family: tuple(grouped[family]) for family in sorted(grouped)}
