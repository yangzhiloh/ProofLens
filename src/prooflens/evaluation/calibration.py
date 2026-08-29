"""Validation-only temperature scaling and operating-point selection."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_curve,
)


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp()

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature


def fit_temperature(logits: Tensor, labels: Tensor) -> TemperatureScaler:
    if logits.ndim != 1 or labels.shape != logits.shape:
        raise ValueError("logits and labels must be matching rank-1 tensors")
    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS([scaler.log_temperature], lr=0.1, max_iter=100)

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(scaler(logits.detach()), labels.detach())
        loss.backward()
        return loss

    optimizer.step(closure)
    return scaler.eval()


@dataclass(frozen=True, slots=True)
class ThresholdReport:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    false_positives: int
    false_negatives: int


def select_operating_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 1 or labels.shape != scores.shape or set(labels.tolist()) != {0, 1}:
        raise ValueError("threshold fitting requires matching scores and both labels")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    objective = true_positive_rate - false_positive_rate
    best = np.flatnonzero(objective == objective.max())
    return float(thresholds[min(best, key=lambda index: abs(thresholds[index] - 0.5))])


def compute_threshold_metrics(scores: np.ndarray, labels: np.ndarray, threshold: float) -> ThresholdReport:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=np.int64)
    if scores.ndim != 1 or labels.shape != scores.shape:
        raise ValueError("scores and labels must have matching one-dimensional shapes")
    predicted = (scores >= threshold).astype(np.int64)
    true_negative, false_positive, false_negative, true_positive = confusion_matrix(
        labels, predicted, labels=[0, 1]
    ).ravel()
    return ThresholdReport(
        threshold=float(threshold), accuracy=float(accuracy_score(labels, predicted)),
        precision=float(precision_score(labels, predicted, zero_division=0)),
        recall=float(recall_score(labels, predicted, zero_division=0)),
        f1=float(f1_score(labels, predicted, zero_division=0)),
        false_positives=int(false_positive), false_negatives=int(false_negative),
    )


def write_calibration(
    scaler: TemperatureScaler,
    threshold_report: ThresholdReport,
    validation_split_hash: str,
    path: Path,
) -> Path:
    payload = {
        "temperature": float(scaler.temperature.detach().cpu()),
        "threshold": float(threshold_report.threshold),
        "operating_point": asdict(threshold_report),
        "validation_split_hash": validation_split_hash,
        "fitted_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
