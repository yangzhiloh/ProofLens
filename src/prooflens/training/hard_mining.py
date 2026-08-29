from __future__ import annotations

import logging
import random
from collections import Counter
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor

from prooflens.data.sampling import stable_seed
from prooflens.data.transforms import TransformSpec, get_spec

LOGGER = logging.getLogger(__name__)


def select_lowest_margin(
    logits_by_condition: Mapping[str, Tensor], labels: Tensor
) -> tuple[str, ...]:
    if not logits_by_condition:
        raise ValueError("at least one condition is required")
    condition_ids = tuple(logits_by_condition)
    stacked = torch.stack(tuple(logits_by_condition.values()), dim=1)
    if labels.ndim != 1 or stacked.shape[0] != labels.shape[0]:
        raise ValueError("condition logits must match the label batch")
    margins = torch.where(labels[:, None] == 1, stacked, -stacked)
    indices = margins.argmin(dim=1).tolist()
    return tuple(condition_ids[index] for index in indices)


class HardTransformMiner:
    def __init__(
        self,
        registry: Sequence[TransformSpec],
        seed: int,
        candidate_count: int = 3,
        exploration_probability: float = 0.20,
    ) -> None:
        grouped: dict[str, list[TransformSpec]] = {}
        for spec in registry:
            grouped.setdefault(spec.family, []).append(spec)
        if not grouped or candidate_count < 1 or candidate_count > len(grouped):
            raise ValueError("candidate_count must fit the available transform families")
        if not 0 <= exploration_probability <= 1:
            raise ValueError("exploration_probability must be between 0 and 1")
        self.by_family = {
            family: tuple(specs) for family, specs in sorted(grouped.items())
        }
        self.seed = seed
        self.candidate_count = candidate_count
        self.exploration_probability = exploration_probability
        self._candidate_families: Counter[str] = Counter()
        self._selected_families: Counter[str] = Counter()

    def sample_candidates(
        self, sample_ids: Sequence[str], epoch: int
    ) -> tuple[tuple[TransformSpec, ...], ...]:
        selections: list[tuple[TransformSpec, ...]] = []
        families = tuple(self.by_family)
        for sample_id in sample_ids:
            rng = random.Random(stable_seed(self.seed, epoch, sample_id, "candidates"))
            chosen = rng.sample(families, k=self.candidate_count)
            selections.append(tuple(rng.choice(self.by_family[name]) for name in chosen))
        return tuple(selections)

    def select(
        self,
        candidate_logits: Tensor,
        candidate_condition_ids: Sequence[Sequence[str]],
        labels: Tensor,
        sample_ids: Sequence[str],
        epoch: int,
    ) -> tuple[str, ...]:
        expected = (len(labels), self.candidate_count)
        if tuple(candidate_logits.shape) != expected:
            raise ValueError(f"candidate logits must have shape {expected}")
        if len(candidate_condition_ids) != len(labels) or len(sample_ids) != len(labels):
            raise ValueError("candidate IDs and sample IDs must match the label batch")
        if any(len(row) != self.candidate_count for row in candidate_condition_ids):
            raise ValueError("each sample must have candidate_count condition IDs")
        margins = torch.where(
            labels[:, None] == 1, candidate_logits, -candidate_logits
        )
        selected_indices = margins.argmin(dim=1).tolist()
        for index, sample_id in enumerate(sample_ids):
            rng = random.Random(stable_seed(self.seed, epoch, sample_id, "explore"))
            if rng.random() < self.exploration_probability:
                selected_indices[index] = rng.randrange(self.candidate_count)
        selected = tuple(
            candidate_condition_ids[row][candidate_index]
            for row, candidate_index in enumerate(selected_indices)
        )
        for row in candidate_condition_ids:
            self._candidate_families.update(get_spec(condition_id).family for condition_id in row)
        self._selected_families.update(get_spec(condition_id).family for condition_id in selected)
        return selected

    def epoch_family_proportions(
        self, *, reset: bool = False
    ) -> dict[str, dict[str, float]]:
        result = {
            "candidate": _proportions(self._candidate_families),
            "selected": _proportions(self._selected_families),
        }
        for family, proportion in result["selected"].items():
            if proportion > 0.60:
                LOGGER.warning(
                    "selected transform family %s exceeds 60 percent: %.1f%%",
                    family,
                    100 * proportion,
                )
        if reset:
            self._candidate_families.clear()
            self._selected_families.clear()
        return result


def _proportions(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    if not total:
        return {}
    return {name: count / total for name, count in sorted(counts.items())}
