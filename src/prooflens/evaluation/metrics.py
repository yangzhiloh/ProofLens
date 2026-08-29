from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from prooflens.data.transforms import group_specs_by_family
from prooflens.errors import MetricPartitionError

_METRIC_COLUMNS = frozenset({"label", "score", "split", "transform_family", "condition_id"})
_ROBUST_FAMILIES = tuple(group_specs_by_family())


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


def compute_condition_auc(frame: pd.DataFrame) -> float:
    """Compute ROC AUC for one valid two-class metric partition."""

    if not isinstance(frame, pd.DataFrame):
        raise MetricPartitionError("metric partition must be a pandas DataFrame")
    missing = {"label", "score"}.difference(frame.columns)
    if missing:
        raise MetricPartitionError(
            f"metric partition is missing required columns: {', '.join(sorted(missing))}"
        )
    if frame.empty:
        raise MetricPartitionError("metric partition must contain rows from both labels")
    labels = pd.to_numeric(frame["label"], errors="coerce").to_numpy(dtype=float)
    scores = pd.to_numeric(frame["score"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not np.isfinite(scores).all():
        raise MetricPartitionError("metric partition labels and scores must be finite")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise MetricPartitionError("metric partition labels must be binary")
    if not np.logical_and(scores >= 0.0, scores <= 1.0).all():
        raise MetricPartitionError("metric partition scores must be probabilities in [0, 1]")
    if np.unique(labels).size != 2:
        raise MetricPartitionError("metric partition must contain both labels")
    return float(roc_auc_score(labels.astype(int), scores))


def compute_metrics(
    predictions: pd.DataFrame,
    evaluation_split: str = "validation",
    generator_split: str = "generator_validation",
) -> MetricReport:
    """Aggregate clean, canonical robustness, and unseen-generator ROC AUC."""

    _validate_metric_input(predictions, evaluation_split, generator_split)
    evaluation = predictions.loc[predictions["split"] == evaluation_split]
    clean = evaluation.loc[evaluation["condition_id"] == "clean"]
    robust = evaluation.loc[evaluation["condition_id"] != "clean"]
    clean_auc = compute_condition_auc(clean)
    if robust.empty:
        raise MetricPartitionError("robust metric partition must contain transformed rows")

    family_counts = robust.groupby("condition_id", sort=True)["transform_family"].nunique()
    ambiguous = tuple(family_counts[family_counts != 1].index.astype(str))
    if ambiguous:
        raise MetricPartitionError(
            "each condition must map to exactly one transform family; invalid conditions: "
            + ", ".join(ambiguous)
        )
    condition_auc = {
        str(condition_id): compute_condition_auc(group)
        for condition_id, group in robust.groupby("condition_id", sort=True)
    }
    condition_family = robust[["condition_id", "transform_family"]].drop_duplicates()
    observed_families = set(condition_family["transform_family"].astype(str))
    missing_families = set(_ROBUST_FAMILIES).difference(observed_families)
    unexpected_families = observed_families.difference(_ROBUST_FAMILIES)
    if missing_families or unexpected_families:
        details: list[str] = []
        if missing_families:
            details.append("missing " + ", ".join(sorted(missing_families)))
        if unexpected_families:
            details.append("unexpected " + ", ".join(sorted(unexpected_families)))
        raise MetricPartitionError(
            "robust metric partition must contain the six canonical transform families ("
            + "; ".join(details)
            + ")"
        )
    family_auc = {
        family: float(
            np.mean(
                [
                    condition_auc[str(condition_id)]
                    for condition_id in condition_family.loc[
                        condition_family["transform_family"] == family, "condition_id"
                    ]
                ]
            )
        )
        for family in _ROBUST_FAMILIES
    }
    macro_robust_auc = float(np.mean(list(family_auc.values())))
    unseen = predictions.loc[
        (predictions["split"] == generator_split) & (predictions["condition_id"] == "clean")
    ]
    unseen_auc = compute_condition_auc(unseen)
    return MetricReport(
        clean_auc=clean_auc,
        condition_auc=condition_auc,
        family_auc=family_auc,
        macro_robust_auc=macro_robust_auc,
        pooled_robust_auc=compute_condition_auc(robust),
        worst_condition_auc=min(condition_auc.values()),
        worst_family_auc=min(family_auc.values()),
        unseen_generator_auc=unseen_auc,
        composite_score=0.5 * clean_auc + 0.5 * macro_robust_auc,
    )


def _validate_metric_input(
    predictions: pd.DataFrame, evaluation_split: str, generator_split: str
) -> None:
    if not isinstance(predictions, pd.DataFrame):
        raise MetricPartitionError("predictions must be a pandas DataFrame")
    missing = _METRIC_COLUMNS.difference(predictions.columns)
    if missing:
        raise MetricPartitionError(
            f"predictions are missing required columns: {', '.join(sorted(missing))}"
        )
    for value, field in ((evaluation_split, "evaluation_split"), (generator_split, "generator_split")):
        if not isinstance(value, str) or not value.strip():
            raise MetricPartitionError(f"{field} must be a nonempty string")
    if evaluation_split == generator_split:
        raise MetricPartitionError("evaluation and generator splits must be independent")
    for column in ("split", "transform_family", "condition_id"):
        values = predictions[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise MetricPartitionError(f"prediction column {column!r} must contain nonempty strings")
    requested = {evaluation_split, generator_split}
    observed = set(predictions["split"].astype(str))
    missing_splits = requested.difference(observed)
    if missing_splits:
        raise MetricPartitionError(
            "predictions are missing requested splits: " + ", ".join(sorted(missing_splits))
        )
    clean_rows = predictions.loc[
        predictions["split"].isin(requested) & (predictions["condition_id"] == "clean")
    ]
    if not clean_rows["transform_family"].eq("clean").all():
        raise MetricPartitionError("clean conditions must use transform_family 'clean'")
    robust_rows = predictions.loc[
        (predictions["split"] == evaluation_split) & (predictions["condition_id"] != "clean")
    ]
    if robust_rows["transform_family"].eq("clean").any():
        raise MetricPartitionError("transformed conditions cannot use transform_family 'clean'")
    for column in ("label", "score"):
        numeric = pd.to_numeric(predictions[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise MetricPartitionError(f"prediction column {column!r} must be finite numeric data")
