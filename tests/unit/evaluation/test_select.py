from __future__ import annotations

import pytest


def test_checkpoint_selection_uses_worst_family_then_unseen_auc() -> None:
    from prooflens.evaluation.select import Candidate, select_best

    candidates = [
        Candidate(
            "a",
            clean_auc=0.9,
            macro_robust_auc=0.8,
            worst_family_auc=0.6,
            unseen_auc=0.8,
        ),
        Candidate(
            "b",
            clean_auc=0.9,
            macro_robust_auc=0.8,
            worst_family_auc=0.7,
            unseen_auc=0.7,
        ),
    ]

    assert select_best(candidates).checkpoint_id == "b"


def test_checkpoint_selection_uses_unseen_then_lower_complexity() -> None:
    from prooflens.evaluation.select import Candidate, rank_candidates

    candidates = [
        Candidate("large", 0.9, 0.8, 0.7, 0.7, parameter_count=200),
        Candidate("unseen", 0.9, 0.8, 0.7, 0.8, parameter_count=300),
        Candidate("small", 0.9, 0.8, 0.7, 0.7, parameter_count=100),
    ]

    assert [candidate.checkpoint_id for candidate in rank_candidates(candidates)] == [
        "unseen",
        "small",
        "large",
    ]


def test_composite_score_has_priority_over_tie_breakers() -> None:
    from prooflens.evaluation.select import Candidate, select_best

    candidates = [
        Candidate("higher-composite", 0.91, 0.81, 0.1, 0.1, parameter_count=999),
        Candidate("better-ties", 0.90, 0.80, 1.0, 1.0, parameter_count=1),
    ]

    assert select_best(candidates).checkpoint_id == "higher-composite"


def test_checkpoint_id_is_a_stable_final_tie_breaker() -> None:
    from prooflens.evaluation.select import Candidate, rank_candidates

    first = Candidate("a", 0.9, 0.8, 0.7, 0.6)
    second = Candidate("b", 0.9, 0.8, 0.7, 0.6)

    assert rank_candidates([first, second]) == (second, first)


def test_selection_rejects_an_empty_candidate_sequence() -> None:
    from prooflens.evaluation.select import rank_candidates, select_best

    with pytest.raises(ValueError, match="at least one"):
        select_best([])
    with pytest.raises(ValueError, match="at least one"):
        rank_candidates([])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("clean_auc", 1.1, "clean_auc"),
        ("macro_robust_auc", float("nan"), "macro_robust_auc"),
        ("worst_family_auc", -0.1, "worst_family_auc"),
        ("unseen_auc", float("inf"), "unseen_auc"),
        ("parameter_count", -1, "parameter_count"),
    ],
)
def test_candidate_rejects_invalid_ranking_values(
    field: str, value: float, message: str
) -> None:
    from prooflens.evaluation.select import Candidate

    values = {
        "checkpoint_id": "checkpoint-a",
        "clean_auc": 0.9,
        "macro_robust_auc": 0.8,
        "worst_family_auc": 0.7,
        "unseen_auc": 0.6,
        "parameter_count": 0,
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        Candidate(**values)
