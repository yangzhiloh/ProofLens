"""Stable prediction records and model-to-manifest evaluation helpers."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import torch

from prooflens.data.collate import PairedBatch
from prooflens.evaluation.metrics import PredictionRecord


PREDICTION_COLUMNS = tuple(PredictionRecord.__dataclass_fields__)


def prediction_records_to_frame(records: Iterable[PredictionRecord]) -> pd.DataFrame:
    return pd.DataFrame([asdict(record) for record in records], columns=PREDICTION_COLUMNS)


def write_prediction_records(records: Iterable[PredictionRecord], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    prediction_records_to_frame(records).to_parquet(path, index=False)
    return path


def read_prediction_records(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def predict_loader(
    model: torch.nn.Module,
    loader,
    *,
    checkpoint_id: str,
    device: torch.device | str = "cpu",
    condition_override: str | None = None,
) -> pd.DataFrame:
    """Predict clean and transformed views from a paired loader.

    A paired loader is intentionally used here so the same path powers training
    smoke tests and full evaluation. The transformed rows carry the collator's
    family and condition metadata; clean rows use ``condition_id='clean'``.
    """

    target_device = torch.device(device)
    model = model.to(target_device).eval()
    records: list[PredictionRecord] = []
    with torch.no_grad():
        for batch in loader:
            if not isinstance(batch, PairedBatch):
                raise TypeError("prediction loader must yield PairedBatch values")
            clean_output = model(batch.clean_pixels.to(target_device))
            transformed_output = model(batch.transformed_pixels.to(target_device))
            clean_logits = clean_output.logits.detach().cpu().tolist()
            transformed_logits = transformed_output.logits.detach().cpu().tolist()
            for index, sample_id in enumerate(batch.sample_ids):
                label = int(batch.labels[index].item())
                clean_logit = float(clean_logits[index])
                transformed_logit = float(transformed_logits[index])
                records.append(_record(
                    sample_id, label, clean_logit, batch.splits[index],
                    batch.generator_families[index], "clean", "clean", checkpoint_id,
                ))
                condition = condition_override or batch.condition_ids[index]
                family = "clean" if condition == "clean" else _family_from_condition(condition)
                records.append(_record(
                    sample_id, label, transformed_logit, batch.splits[index],
                    batch.generator_families[index], family, condition, checkpoint_id,
                ))
    return prediction_records_to_frame(records)


def _record(sample_id, label, logit, split, generator, transform_family, condition, checkpoint):
    return PredictionRecord(
        sample_id=str(sample_id), label=int(label), logit=float(logit),
        score=float(torch.sigmoid(torch.tensor(logit))), split=str(split),
        generator_family=str(generator), transform_family=str(transform_family),
        condition_id=str(condition), checkpoint_id=str(checkpoint),
    )


def _family_from_condition(condition_id: str) -> str:
    if condition_id.startswith("color_jitter"):
        return "color_jitter"
    if condition_id.startswith("center_crop"):
        return "center_crop"
    return condition_id.split("_", 1)[0]
