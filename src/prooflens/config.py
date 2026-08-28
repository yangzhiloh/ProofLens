from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(StrictModel):
    manifest: Path


class ModelConfig(StrictModel):
    name: str = "facebook/dinov2-base"
    stage: Literal["head", "last2"] = "head"


class TrainingConfig(StrictModel):
    epochs: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    learning_rate: float = Field(default=1e-4, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    max_gradient_norm: float = Field(default=1.0, gt=0)
    warmup_fraction: float = Field(default=0.10, ge=0, lt=1)
    early_stopping_patience: int = Field(default=2, ge=1)
    num_workers: int = Field(default=0, ge=0)


class TransformConfig(StrictModel):
    enabled: bool = False
    hard_mining: bool = False
    candidate_count: int = Field(default=3, ge=1, le=6)
    exploration_probability: float = Field(default=0.20, ge=0, le=1)


class LossConfig(StrictModel):
    clean_bce: float = Field(default=1.0, ge=0)
    transformed_bce: float = Field(default=0.0, ge=0)
    prediction_consistency: float = Field(default=0.0, ge=0)
    feature_consistency: float = Field(default=0.0, ge=0)


class ExperimentConfig(StrictModel):
    seed: int
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    transforms: TransformConfig = Field(default_factory=TransformConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    output_dir: Path

    def resolve(self, base: Path) -> "ExperimentConfig":
        raw = self.model_dump()
        raw["data"]["manifest"] = base / self.data.manifest
        raw["output_dir"] = base / self.output_dir
        return ExperimentConfig.model_validate(raw)


def load_config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
