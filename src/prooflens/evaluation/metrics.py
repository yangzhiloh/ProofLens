"""Partition-safe robustness metrics and checkpoint ranking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from prooflens.errors import MetricPartitionError


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    sample_id: str
    label: int
    logit: float
    score: float
    split: str
    generator_family: str
    transform_family: str
    condition_id: str
    checkpoint_id: str


@dataclass(frozen=True, slots=True)
class MetricReport:
    clean_auc: float
    condition_auc: dict[str, float]
    family_auc: dict[str, float]
    macro_robust_auc: float
    pooled_robust_auc: float
    worst_condition_auc: float
    worst_family_auc: float
    unseen_generator_auc: float
    composite_score: float
    model_parameters: int = 0
    inference_ms_median: float = float("nan")


@dataclass(frozen=True, slots=True)
class Candidate:
    checkpoint_id: str
    clean_auc: float
    macro_robust_auc: float
    worst_family_auc: float
    unseen_auc: float
    parameter_count: int = 0

    @property
    def composite_score(self) -> float:
        return 0.5 * self.clean_auc + 0.5 * self.macro_robust_auc


def compute_condition_auc(frame: pd.DataFrame) -> float:
    if not isinstance(frame, pd.DataFrame) or not {"label", "score"}.issubset(frame.columns):
        raise MetricPartitionError("condition metrics require label and score columns")
    return _auc(frame)


def compute_metrics(
    predictions: pd.DataFrame,
    evaluation_split: str = "validation",
    generator_split: str = "generator_validation",
) -> MetricReport:
    _validate_prediction_frame(predictions)
    evaluation = predictions[predictions["split"] == evaluation_split]
    clean = evaluation[evaluation["condition_id"] == "clean"]
    clean_auc = _auc(clean)
    robust = evaluation[evaluation["condition_id"] != "clean"]
    if robust.empty:
        raise MetricPartitionError("robust evaluation must contain at least one condition")
    condition_auc = {
        str(name): _auc(group)
        for name, group in robust.groupby("condition_id", sort=True)
    }
    family_auc: dict[str, float] = {}
    for family, group in robust.groupby("transform_family", sort=True):
        condition_names = group["condition_id"].drop_duplicates().tolist()
        family_auc[str(family)] = float(np.mean([condition_auc[name] for name in condition_names]))
    if len(family_auc) != 6:
        raise MetricPartitionError(
            f"robust evaluation must contain all six transform families; found {sorted(family_auc)}"
        )
    unseen = predictions[
        (predictions["split"] == generator_split) & (predictions["condition_id"] == "clean")
    ]
    unseen_auc = _auc(unseen)
    macro = float(np.mean(list(family_auc.values())))
    return MetricReport(
        clean_auc=clean_auc,
        condition_auc=condition_auc,
        family_auc=family_auc,
        macro_robust_auc=macro,
        pooled_robust_auc=_auc(robust),
        worst_condition_auc=min(condition_auc.values()),
        worst_family_auc=min(family_auc.values()),
        unseen_generator_auc=unseen_auc,
        composite_score=0.5 * clean_auc + 0.5 * macro,
    )


def select_best(candidates: Sequence[Candidate]) -> Candidate:
    if not candidates:
        raise ValueError("at least one checkpoint candidate is required")
    return max(
        candidates,
        key=lambda item: (
            item.composite_score,
            item.worst_family_auc,
            item.unseen_auc,
            -item.parameter_count,
            item.checkpoint_id,
        ),
    )


def _auc(frame: pd.DataFrame) -> float:
    if frame.empty or frame["label"].nunique() != 2:
        raise MetricPartitionError("metric partition must contain both labels")
    try:
        return float(roc_auc_score(frame["label"], frame["score"]))
    except ValueError as error:
        raise MetricPartitionError(f"could not compute ROC AUC: {error}") from error


def _validate_prediction_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise MetricPartitionError("predictions must be a pandas DataFrame")
    missing = set(PREDICTION_FIELDS) - set(frame.columns)
    if missing:
        raise MetricPartitionError(f"predictions are missing required fields: {sorted(missing)}")


PREDICTION_FIELDS = (
    "sample_id", "label", "logit", "score", "split", "generator_family",
    "transform_family", "condition_id", "checkpoint_id",
)
