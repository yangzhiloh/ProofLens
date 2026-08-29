from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

import pandas as pd
import torch
from torch import Tensor, nn

from prooflens.data.collate import PairedBatch
from prooflens.data.transforms import get_spec
from prooflens.errors import MetricPartitionError

PREDICTION_COLUMNS = (
    "sample_id",
    "label",
    "logit",
    "score",
    "split",
    "generator_family",
    "transform_family",
    "condition_id",
    "checkpoint_id",
)
_TRANSFORM_FAMILIES = frozenset(
    {"clean", "jpeg", "blur", "resize", "noise", "color_jitter", "center_crop"}
)


@dataclass(frozen=True, slots=True)
class PredictionRecord:
    """One model prediction with enough provenance for robust evaluation."""

    sample_id: str
    label: int
    logit: float
    score: float
    split: str
    generator_family: str
    transform_family: str
    condition_id: str
    checkpoint_id: str

    def __post_init__(self) -> None:
        for field in (
            "sample_id",
            "split",
            "generator_family",
            "transform_family",
            "condition_id",
            "checkpoint_id",
        ):
            _require_text(getattr(self, field), field)
        if (
            not isinstance(self.label, int)
            or isinstance(self.label, bool)
            or self.label not in (0, 1)
        ):
            raise MetricPartitionError("prediction label must be a binary integer")
        if not isinstance(self.logit, (int, float)) or isinstance(self.logit, bool):
            raise MetricPartitionError("prediction logit must be a finite number")
        if not math.isfinite(float(self.logit)):
            raise MetricPartitionError("prediction logit must be a finite number")
        if not isinstance(self.score, (int, float)) or isinstance(self.score, bool):
            raise MetricPartitionError("prediction score must be a finite probability")
        if not math.isfinite(float(self.score)) or not 0.0 <= float(self.score) <= 1.0:
            raise MetricPartitionError("prediction score must be a finite probability")
        expected_score = sigmoid(float(self.logit))
        if not math.isclose(float(self.score), expected_score, rel_tol=1e-12, abs_tol=1e-12):
            raise MetricPartitionError("prediction score must equal the sigmoid of its logit")
        if self.transform_family not in _TRANSFORM_FAMILIES:
            raise MetricPartitionError(
                f"unsupported prediction transform family: {self.transform_family!r}"
            )
        if (self.condition_id == "clean") != (self.transform_family == "clean"):
            raise MetricPartitionError(
                "clean predictions must use condition_id and transform_family 'clean' together"
            )
        object.__setattr__(self, "logit", float(self.logit))
        object.__setattr__(self, "score", float(self.score))

    @classmethod
    def from_logit(
        cls,
        *,
        sample_id: str,
        label: int,
        logit: float,
        split: str,
        generator_family: str,
        transform_family: str,
        condition_id: str,
        checkpoint_id: str,
    ) -> PredictionRecord:
        """Create a record while deriving the probability from the raw logit."""

        numeric_logit = float(logit)
        return cls(
            sample_id=sample_id,
            label=label,
            logit=numeric_logit,
            score=sigmoid(numeric_logit),
            split=split,
            generator_family=generator_family,
            transform_family=transform_family,
            condition_id=condition_id,
            checkpoint_id=checkpoint_id,
        )


def sigmoid(logit: float) -> float:
    """Compute a numerically stable scalar sigmoid."""

    if not math.isfinite(logit):
        raise MetricPartitionError("prediction logit must be a finite number")
    if logit >= 0.0:
        inverse = math.exp(-logit)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(logit)
    return exponential / (1.0 + exponential)


def records_to_frame(records: Sequence[PredictionRecord]) -> pd.DataFrame:
    """Serialize validated records in a stable declaration-order schema."""

    validated: list[PredictionRecord] = []
    for position, record in enumerate(records):
        if not isinstance(record, PredictionRecord):
            raise MetricPartitionError(
                f"prediction at position {position} must be a PredictionRecord"
            )
        validated.append(record)
    return pd.DataFrame([asdict(record) for record in validated], columns=PREDICTION_COLUMNS)


def write_predictions(records: Sequence[PredictionRecord], output_path: Path) -> Path:
    """Write prediction records to Parquet without exposing a partial destination."""

    output_path = Path(output_path)
    if output_path.suffix.lower() != ".parquet":
        raise MetricPartitionError("prediction output path must use the .parquet suffix")
    frame = records_to_frame(records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary_path, index=False)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def predict_loader(
    model: nn.Module,
    loader: Iterable[PairedBatch],
    *,
    checkpoint_id: str,
    device: str | torch.device = "cpu",
    condition_override: str | None = None,
) -> pd.DataFrame:
    """Predict clean and transformed rows from paired evaluation batches."""

    if not isinstance(model, nn.Module):
        raise TypeError("prediction model must be a torch.nn.Module")
    target_device = torch.device(device)
    model = model.to(target_device).eval()
    records: list[PredictionRecord] = []
    with torch.inference_mode():
        for batch in loader:
            if not isinstance(batch, PairedBatch):
                raise TypeError("prediction loader must yield PairedBatch values")
            batch_size = len(batch.sample_ids)
            clean_logits = _batch_logits(model(batch.clean_pixels.to(target_device)), batch_size)
            transformed_logits = _batch_logits(
                model(batch.transformed_pixels.to(target_device)), batch_size
            )
            for index, sample_id in enumerate(batch.sample_ids):
                common = {
                    "sample_id": sample_id,
                    "label": int(batch.labels[index].item()),
                    "split": batch.splits[index],
                    "generator_family": batch.generator_families[index],
                    "checkpoint_id": checkpoint_id,
                }
                records.append(
                    PredictionRecord.from_logit(
                        **common,
                        logit=float(clean_logits[index]),
                        transform_family="clean",
                        condition_id="clean",
                    )
                )
                condition_id = condition_override or batch.condition_ids[index]
                transform_family = (
                    "clean" if condition_id == "clean" else get_spec(condition_id).family
                )
                records.append(
                    PredictionRecord.from_logit(
                        **common,
                        logit=float(transformed_logits[index]),
                        transform_family=transform_family,
                        condition_id=condition_id,
                    )
                )
    return records_to_frame(records)


def _batch_logits(output: object, batch_size: int) -> Tensor:
    logits = output if isinstance(output, Tensor) else getattr(output, "logits", None)
    if not isinstance(logits, Tensor) or logits.shape != (batch_size,):
        raise MetricPartitionError("prediction model must return one logit per image")
    if not logits.is_floating_point() or not torch.isfinite(logits).all().item():
        raise MetricPartitionError("prediction model logits must be finite floating values")
    return logits.detach().to(device="cpu", dtype=torch.float32)


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MetricPartitionError(f"prediction {field} must be a nonempty string")
    return value
