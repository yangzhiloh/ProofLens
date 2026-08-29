from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from prooflens.config import ExperimentConfig
from prooflens.data.collate import PairedBatch, PairedBatchCollator
from prooflens.data.dataset import SourceImageDataset
from prooflens.data.sampling import FamilyBalancedTransformSampler, make_weighted_sampler
from prooflens.data.transforms import canonical_specs
from prooflens.errors import TrainingError
from prooflens.inference.preprocess import create_dinov2_processor
from prooflens.models.detector import DinoDetector
from prooflens.training.checkpoints import CheckpointManager
from prooflens.training.hard_mining import (
    HardMiningCollator,
    HardTransformMiner,
    compute_hard_mined_loss,
)
from prooflens.training.losses import SurvivalLossWeights, compute_survival_loss
from prooflens.training.run_metadata import (
    collect_run_metadata,
    write_run_metadata,
)


class EpochCollator(Protocol):
    def set_epoch(self, epoch: int) -> None: ...


@dataclass(frozen=True, slots=True)
class ValidationSnapshot:
    clean_auc: float
    macro_robust_auc: float
    composite_score: float
    metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    output_dir: Path
    best_checkpoint: Path
    best_composite_score: float


@dataclass(frozen=True, slots=True)
class TrainingComponents:
    model: nn.Module
    train_loader: Iterable[Any]
    optimizer: Optimizer
    scheduler: Any
    collator: EpochCollator
    batch_loss_callback: Callable[[nn.Module, Any], Tensor]
    validation_callback: Callable[[nn.Module, int], ValidationSnapshot]
    device: str
    epoch_metrics_callback: Callable[[], Mapping[str, float]] | None = None


def run_training(
    config: ExperimentConfig,
    *,
    components: TrainingComponents | None = None,
    resume_from: Path | None = None,
) -> TrainingResult:
    resolved_components = components or build_training_components(config)
    trainer = Trainer(config=config, components=resolved_components)
    if resume_from is not None:
        trainer.resume(resume_from)
    return trainer.fit()


class Trainer:
    def __init__(self, config: ExperimentConfig, components: TrainingComponents) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = components.model
        self.train_loader = components.train_loader
        self.optimizer = components.optimizer
        self.scheduler = components.scheduler
        self.collator = components.collator
        self.batch_loss_callback = components.batch_loss_callback
        self.validation_callback = components.validation_callback
        self.epoch_metrics_callback = components.epoch_metrics_callback
        self.device = components.device
        self.checkpoints = CheckpointManager(self.output_dir / "checkpoints")
        self.global_step = 0
        self.start_epoch = 1
        self.amp_enabled = self.device.startswith("cuda") and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp_enabled)

    def resume(self, checkpoint: Path) -> None:
        state = self.checkpoints.load(
            checkpoint,
            self.model,
            self.optimizer,
            self.scheduler,
        )
        self.start_epoch = state.epoch + 1
        self.global_step = state.global_step

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.collator.set_epoch(epoch)
        running_loss = 0.0
        batch_count = 0
        self.optimizer.zero_grad(set_to_none=True)
        accumulation = self.config.training.gradient_accumulation_steps
        try:
            loader_length = len(self.train_loader)  # type: ignore[arg-type]
        except TypeError:
            loader_length = None
        for step, batch in enumerate(self.train_loader, start=1):
            try:
                with torch.autocast(device_type="cuda", enabled=self.amp_enabled):
                    raw_loss = self.batch_loss_callback(self.model, batch)
                self.scaler.scale(raw_loss / accumulation).backward()
            except torch.cuda.OutOfMemoryError as error:
                recommended = max(1, self.config.training.batch_size // 2)
                raise TrainingError(
                    "CUDA out of memory at batch size "
                    f"{self.config.training.batch_size}; retry with batch size {recommended}"
                ) from error
            running_loss += float(raw_loss.detach())
            batch_count += 1
            should_step = step % accumulation == 0 or (
                loader_length is not None and step == loader_length
            )
            if should_step:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.max_gradient_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
        if batch_count and loader_length is None and batch_count % accumulation:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.max_gradient_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.global_step += 1
        if not batch_count:
            raise TrainingError("training loader produced no batches")
        return running_loss / batch_count

    def fit(self) -> TrainingResult:
        if self.start_epoch == 1:
            _seed_everything(self.config.seed)
        metadata = collect_run_metadata(self.config, device=self.device)
        write_run_metadata(metadata, self.output_dir / "run_metadata.json")
        best_score = -math.inf
        best_checkpoint: Path | None = None
        epochs_without_improvement = 0
        for epoch in range(self.start_epoch, self.config.training.epochs + 1):
            train_loss = self.train_epoch(epoch)
            training_metrics = (
                dict(self.epoch_metrics_callback())
                if self.epoch_metrics_callback is not None
                else {}
            )
            validation = self.validation_callback(self.model, epoch)
            checkpoint = self.checkpoints.save_epoch(
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                self.global_step,
                metadata.config_sha256,
            )
            _append_history(
                self.output_dir,
                epoch,
                train_loss,
                validation,
                training_metrics,
            )
            if validation.composite_score > best_score:
                best_score = validation.composite_score
                best_checkpoint = self.checkpoints.mark_best(checkpoint)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.config.training.early_stopping_patience:
                    break
        if best_checkpoint is None:
            raise TrainingError("training produced no checkpoint")
        return TrainingResult(self.output_dir, best_checkpoint, best_score)


def build_training_components(config: ExperimentConfig) -> TrainingComponents:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frame = pd.read_parquet(config.data.manifest)
    train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    validation_frame = frame.loc[frame["split"] == "validation"].reset_index(drop=True)
    if train_frame.empty or validation_frame.empty:
        raise TrainingError("assigned manifest requires train and validation rows")
    processor = create_dinov2_processor()
    hard_miner: HardTransformMiner | None = None
    if config.transforms.hard_mining:
        hard_miner = HardTransformMiner(
            canonical_specs(),
            seed=config.seed,
            candidate_count=config.transforms.candidate_count,
            exploration_probability=config.transforms.exploration_probability,
        )
        train_collator: Any = HardMiningCollator(processor=processor)
    else:
        train_collator = PairedBatchCollator(
            processor=processor,
            sampler=FamilyBalancedTransformSampler(),
            seed=config.seed,
        )
    validation_collator = PairedBatchCollator(
        processor=processor,
        sampler=FamilyBalancedTransformSampler(),
        seed=config.seed,
    )
    train_loader = DataLoader(
        SourceImageDataset(train_frame),
        batch_size=config.training.batch_size,
        sampler=make_weighted_sampler(train_frame, seed=config.seed),
        num_workers=config.training.num_workers,
        collate_fn=train_collator,
    )
    validation_loader = DataLoader(
        SourceImageDataset(validation_frame),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=validation_collator,
    )
    model = DinoDetector.from_pretrained(config.model.name).to(device)
    model.set_trainable_stage(config.model.stage)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    updates_per_epoch = max(
        1,
        math.ceil(len(train_loader) / config.training.gradient_accumulation_steps),
    )
    total_updates = updates_per_epoch * config.training.epochs
    warmup_updates = int(total_updates * config.training.warmup_fraction)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: _learning_rate_multiplier(step, warmup_updates, total_updates),
    )
    weights = SurvivalLossWeights(
        clean_bce=config.loss.clean_bce,
        transformed_bce=config.loss.transformed_bce,
        prediction_consistency=config.loss.prediction_consistency,
        feature_consistency=config.loss.feature_consistency,
    )

    if config.transforms.hard_mining:
        if hard_miner is None:  # pragma: no cover - guarded by the shared condition
            raise TrainingError("hard-mining configuration failed to create a miner")

        def batch_loss(current_model: nn.Module, batch: Any) -> Tensor:
            return compute_hard_mined_loss(
                model=current_model,
                batch=batch,
                processor=processor,
                miner=hard_miner,
                epoch=train_collator.epoch,
                device=device,
                weights=weights,
            ).total

    else:

        def batch_loss(current_model: nn.Module, batch: PairedBatch) -> Tensor:
            labels = batch.labels.to(device)
            clean = current_model(batch.clean_pixels.to(device))
            transformed = current_model(batch.transformed_pixels.to(device))
            return compute_survival_loss(clean, transformed, labels, weights).total

    def epoch_metrics() -> Mapping[str, float]:
        if hard_miner is None:
            return {}
        summary = hard_miner.epoch_family_proportions(reset=True)
        return {
            f"{kind}/{family}": proportion
            for kind, proportions in summary.items()
            for family, proportion in proportions.items()
        }

    def validate(current_model: nn.Module, epoch: int) -> ValidationSnapshot:
        validation_collator.set_epoch(epoch)
        current_model.eval()
        labels: list[float] = []
        clean_scores: list[float] = []
        robust_scores: list[float] = []
        with torch.no_grad():
            for batch in validation_loader:
                labels.extend(batch.labels.tolist())
                clean_scores.extend(
                    current_model(batch.clean_pixels.to(device)).logits.cpu().tolist()
                )
                robust_scores.extend(
                    current_model(batch.transformed_pixels.to(device)).logits.cpu().tolist()
                )
        try:
            clean_auc = float(roc_auc_score(labels, clean_scores))
            robust_auc = float(roc_auc_score(labels, robust_scores))
        except ValueError as error:
            raise TrainingError("validation requires both binary labels") from error
        composite = 0.5 * clean_auc + 0.5 * robust_auc
        return ValidationSnapshot(
            clean_auc,
            robust_auc,
            composite,
            {"clean_auc": clean_auc, "macro_robust_auc": robust_auc},
        )

    return TrainingComponents(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        collator=train_collator,
        batch_loss_callback=batch_loss,
        validation_callback=validate,
        device=device,
        epoch_metrics_callback=epoch_metrics,
    )


def _learning_rate_multiplier(step: int, warmup: int, total: int) -> float:
    if warmup and step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _append_history(
    output_dir: Path,
    epoch: int,
    train_loss: float,
    validation: ValidationSnapshot,
    training_metrics: Mapping[str, float],
) -> None:
    record = {
        "epoch": epoch,
        "train_loss": train_loss,
        "clean_auc": validation.clean_auc,
        "macro_robust_auc": validation.macro_robust_auc,
        "composite_score": validation.composite_score,
        "metrics": dict(validation.metrics),
        "training_metrics": dict(training_metrics),
    }
    with (output_dir / "history.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
