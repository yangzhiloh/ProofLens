from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from torch import Tensor, nn

from prooflens.errors import MetricPartitionError


class TemperatureScaler(nn.Module):
    """A positive scalar temperature represented in unconstrained log space."""

    def __init__(self, *, dtype: torch.dtype = torch.float32, device: torch.device | None = None):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros((), dtype=dtype, device=device))

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp()

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature


@dataclass(frozen=True, slots=True)
class ThresholdReport:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positives: int
    false_negatives: int


def fit_temperature(logits: Tensor, labels: Tensor) -> TemperatureScaler:
    """Fit one temperature on detached validation logits using LBFGS."""

    validated_logits, validated_labels = _validated_calibration_tensors(logits, labels)
    scaler = TemperatureScaler(dtype=validated_logits.dtype, device=validated_logits.device)
    optimizer = torch.optim.LBFGS([scaler.log_temperature], lr=0.1, max_iter=100)
    before = F.binary_cross_entropy_with_logits(validated_logits, validated_labels).detach()

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(
            scaler(validated_logits), validated_labels
        )
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        scaler.log_temperature.clamp_(min=-20.0, max=20.0)
        after = F.binary_cross_entropy_with_logits(
            scaler(validated_logits), validated_labels
        )
        if not torch.isfinite(after) or after > before + 1e-7:
            scaler.log_temperature.zero_()
    return scaler.eval()


def select_operating_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    """Maximize Youden's J on validation scores with a stable 0.5 tie-break."""

    validated_scores, validated_labels = _validated_threshold_arrays(scores, labels)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        validated_labels, validated_scores
    )
    objective = true_positive_rate - false_positive_rate
    best_value = objective.max()
    best = np.flatnonzero(np.isclose(objective, best_value, rtol=0.0, atol=1e-12))
    finite_best = [index for index in best if math.isfinite(float(thresholds[index]))]
    if not finite_best:
        raise MetricPartitionError("validation scores did not produce a finite threshold")
    index = min(
        finite_best,
        key=lambda candidate: (abs(float(thresholds[candidate]) - 0.5), candidate),
    )
    return float(thresholds[index])


def compute_threshold_metrics(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> ThresholdReport:
    """Compute threshold-dependent binary metrics from calibrated probabilities."""

    validated_scores, validated_labels = _validated_threshold_arrays(scores, labels)
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise MetricPartitionError("operating threshold must be a finite probability")
    numeric_threshold = float(threshold)
    predicted = (validated_scores >= numeric_threshold).astype(np.int64)
    _, false_positive, false_negative, _ = confusion_matrix(
        validated_labels, predicted, labels=[0, 1]
    ).ravel()
    return ThresholdReport(
        threshold=numeric_threshold,
        accuracy=float(accuracy_score(validated_labels, predicted)),
        precision=float(precision_score(validated_labels, predicted, zero_division=0)),
        recall=float(recall_score(validated_labels, predicted, zero_division=0)),
        f1=float(f1_score(validated_labels, predicted, zero_division=0)),
        false_positives=int(false_positive),
        false_negatives=int(false_negative),
    )


def write_calibration(
    *,
    temperature: float,
    threshold: float,
    validation_split_hash: str,
    path: Path,
    fitted_at: str | None = None,
) -> Path:
    """Persist calibration parameters with their validation-only provenance."""

    numeric_temperature = _positive_finite(temperature, "temperature")
    numeric_threshold = _probability(threshold, "threshold")
    if not isinstance(validation_split_hash, str) or not validation_split_hash.strip():
        raise MetricPartitionError("validation_split_hash must be a nonempty string")
    timestamp = fitted_at or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise MetricPartitionError("fitted_at must be a nonempty timestamp")
    payload = {
        "fitted_at": timestamp,
        "temperature": numeric_temperature,
        "threshold": numeric_threshold,
        "validation_split_hash": validation_split_hash,
    }
    _atomic_write_text(
        Path(path), json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return Path(path)


def threshold_report_payload(report: ThresholdReport) -> dict[str, float | int]:
    """Return a JSON-ready threshold report while retaining the typed public object."""

    if not isinstance(report, ThresholdReport):
        raise TypeError("report must be a ThresholdReport")
    return asdict(report)


def _validated_calibration_tensors(logits: Tensor, labels: Tensor) -> tuple[Tensor, Tensor]:
    if not isinstance(logits, Tensor) or not isinstance(labels, Tensor):
        raise MetricPartitionError("calibration logits and labels must be Torch tensors")
    if logits.ndim != 1 or labels.ndim != 1:
        raise MetricPartitionError("calibration logits and labels must be one-dimensional")
    if logits.numel() != labels.numel():
        raise MetricPartitionError("calibration logits and labels must have the same length")
    if logits.numel() == 0:
        raise MetricPartitionError("calibration requires nonempty validation tensors")
    if not logits.is_floating_point():
        raise MetricPartitionError("calibration logits must use a floating dtype")
    detached_logits = logits.detach().clone()
    detached_labels = labels.detach().to(device=logits.device, dtype=logits.dtype).clone()
    if not torch.isfinite(detached_logits).all() or not torch.isfinite(detached_labels).all():
        raise MetricPartitionError("calibration logits and labels must be finite")
    if not torch.logical_or(detached_labels == 0.0, detached_labels == 1.0).all():
        raise MetricPartitionError("calibration labels must be binary")
    if torch.unique(detached_labels).numel() != 2:
        raise MetricPartitionError("calibration labels must contain both labels")
    return detached_logits, detached_labels


def _validated_threshold_arrays(
    scores: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    try:
        numeric_scores = np.asarray(scores, dtype=np.float64)
        numeric_labels = np.asarray(labels, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise MetricPartitionError("threshold scores and labels must be numeric arrays") from error
    if numeric_scores.ndim != 1 or numeric_labels.ndim != 1:
        raise MetricPartitionError("threshold scores and labels must be one-dimensional")
    if numeric_scores.size != numeric_labels.size:
        raise MetricPartitionError("threshold scores and labels must have the same length")
    if numeric_scores.size == 0:
        raise MetricPartitionError("threshold fitting requires nonempty validation arrays")
    if not np.isfinite(numeric_scores).all() or not np.isfinite(numeric_labels).all():
        raise MetricPartitionError("threshold scores and labels must be finite")
    if not np.logical_and(numeric_scores >= 0.0, numeric_scores <= 1.0).all():
        raise MetricPartitionError("threshold scores must be probabilities in [0, 1]")
    if not np.isin(numeric_labels, (0.0, 1.0)).all():
        raise MetricPartitionError("threshold labels must be binary")
    if np.unique(numeric_labels).size != 2:
        raise MetricPartitionError("threshold fitting requires both labels")
    return numeric_scores, numeric_labels.astype(np.int64)


def _positive_finite(value: float, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise MetricPartitionError(f"{field} must be a finite positive number")
    return float(value)


def _probability(value: float, field: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise MetricPartitionError(f"{field} must be a finite probability")
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise MetricPartitionError(f"{field} must be a probability in [0, 1]")
    return numeric


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(text, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
