from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Candidate:
    checkpoint_id: str
    clean_auc: float
    macro_robust_auc: float
    worst_family_auc: float
    unseen_auc: float
    parameter_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint_id, str) or not self.checkpoint_id.strip():
            raise ValueError("checkpoint_id must be a nonempty string")
        for field in ("clean_auc", "macro_robust_auc", "worst_family_auc", "unseen_auc"):
            value = getattr(self, field)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{field} must be a finite value in [0, 1]")
            numeric = float(value)
            if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
                raise ValueError(f"{field} must be a finite value in [0, 1]")
            object.__setattr__(self, field, numeric)
        if (
            not isinstance(self.parameter_count, int)
            or isinstance(self.parameter_count, bool)
            or self.parameter_count < 0
        ):
            raise ValueError("parameter_count must be a nonnegative integer")

    @property
    def composite_score(self) -> float:
        return 0.5 * self.clean_auc + 0.5 * self.macro_robust_auc


def rank_candidates(candidates: Sequence[Candidate]) -> tuple[Candidate, ...]:
    """Rank checkpoints by the approved score and deterministic tie-breakers."""

    if not candidates:
        raise ValueError("at least one checkpoint candidate is required")
    for position, candidate in enumerate(candidates):
        if not isinstance(candidate, Candidate):
            raise TypeError(f"candidate at position {position} must be a Candidate")
    return tuple(sorted(candidates, key=_ranking_key, reverse=True))


def select_best(candidates: Sequence[Candidate]) -> Candidate:
    """Return the leading checkpoint without consulting final-test results."""

    return rank_candidates(candidates)[0]


def _ranking_key(candidate: Candidate) -> tuple[float, float, float, int, str]:
    return (
        candidate.composite_score,
        candidate.worst_family_auc,
        candidate.unseen_auc,
        -candidate.parameter_count,
        candidate.checkpoint_id,
    )
