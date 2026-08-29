"""Supplemental redistribution stress evaluation, intentionally separate from ranking."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from prooflens.data.dataset import SourceItem
from prooflens.data.sampling import stable_seed
from prooflens.data.stress_transforms import apply_stress_transform, stress_specs
from prooflens.errors import MetricPartitionError
from prooflens.evaluation.metrics import compute_condition_auc
from prooflens.evaluation.predict import sigmoid

STRESS_PREDICTION_COLUMNS = (
    "sample_id",
    "label",
    "logit",
    "score",
    "clean_score",
    "split",
    "generator_family",
    "condition_id",
    "checkpoint_id",
    "transform_metadata",
)
_CONDITION_IDS = tuple(spec.condition_id for spec in stress_specs())


def evaluate_stress(
    items: Iterable[SourceItem], backend: object, *, checkpoint_id: str, seed: int
) -> pd.DataFrame:
    """Return one secondary-condition prediction per source image and condition."""

    predict_logit = getattr(backend, "predict_logit", None)
    if not callable(predict_logit):
        raise TypeError("stress backend must provide predict_logit")
    records: list[dict[str, object]] = []
    for item in items:
        if not isinstance(item, SourceItem):
            raise TypeError("stress evaluation items must be SourceItem values")
        clean_logit = float(predict_logit(item.image))
        _finite(clean_logit, "clean logit")
        for spec in stress_specs():
            result = apply_stress_transform(
                item.image,
                spec,
                seed=stable_seed(seed, item.sample_id, spec.condition_id),
            )
            logit = float(predict_logit(result.image))
            _finite(logit, "stress logit")
            records.append(
                {
                    "sample_id": item.sample_id,
                    "label": item.label,
                    "logit": logit,
                    "score": sigmoid(logit),
                    "clean_score": sigmoid(clean_logit),
                    "split": item.split,
                    "generator_family": item.generator_family,
                    "condition_id": spec.condition_id,
                    "checkpoint_id": checkpoint_id,
                    "transform_metadata": json.dumps(result.metadata, sort_keys=True),
                }
            )
    return pd.DataFrame(records, columns=STRESS_PREDICTION_COLUMNS)


def compute_stress_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    """Compute supplemental AUC and clean-to-stress probability shifts by condition."""

    _validate_stress_predictions(predictions)
    conditions: dict[str, dict[str, object]] = {}
    for condition_id in _CONDITION_IDS:
        partition = predictions.loc[predictions["condition_id"] == condition_id]
        shift = partition["score"].to_numpy(dtype=float) - partition["clean_score"].to_numpy(
            dtype=float
        )
        conditions[condition_id] = {
            "auc": compute_condition_auc(partition),
            "probability_shift": {
                "mean": float(np.mean(shift)),
                "median": float(np.median(shift)),
                "mean_absolute": float(np.mean(np.abs(shift))),
            },
        }
    return {"conditions": conditions}


def write_stress_predictions(predictions: pd.DataFrame, output_path: Path) -> Path:
    """Write a complete supplemental prediction artifact atomically."""

    _validate_stress_predictions(predictions)
    destination = Path(output_path)
    if destination.suffix.lower() != ".parquet":
        raise MetricPartitionError("stress prediction output path must use the .parquet suffix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        predictions.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _validate_stress_predictions(predictions: pd.DataFrame) -> None:
    if not isinstance(predictions, pd.DataFrame):
        raise MetricPartitionError("stress predictions must be a pandas DataFrame")
    missing = set(STRESS_PREDICTION_COLUMNS).difference(predictions.columns)
    if missing:
        raise MetricPartitionError(
            "stress predictions are missing required columns: " + ", ".join(sorted(missing))
        )
    if predictions.empty:
        raise MetricPartitionError("stress predictions must contain rows")
    observed = tuple(predictions["condition_id"].drop_duplicates())
    if set(observed) != set(_CONDITION_IDS):
        raise MetricPartitionError("stress predictions must contain exactly the four stress conditions")
    for column in ("label", "score", "clean_score", "logit"):
        values = pd.to_numeric(predictions[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise MetricPartitionError(f"stress prediction column {column!r} must be finite")
    labels = predictions["label"].to_numpy(dtype=float)
    if not np.isin(labels, (0.0, 1.0)).all():
        raise MetricPartitionError("stress prediction labels must be binary")
    for column in ("score", "clean_score"):
        values = predictions[column].to_numpy(dtype=float)
        if not np.logical_and(values >= 0.0, values <= 1.0).all():
            raise MetricPartitionError(f"stress prediction column {column!r} must be probabilities")


def _finite(value: float, field: str) -> None:
    if not math.isfinite(value):
        raise MetricPartitionError(f"stress {field} must be finite")
