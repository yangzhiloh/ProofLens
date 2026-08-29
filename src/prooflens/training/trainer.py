"""CPU-safe training loop used by the experiment CLI and miniature workflow."""

from __future__ import annotations

import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Any

import numpy as np
import pandas as pd
import yaml
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from prooflens.config import ExperimentConfig
from prooflens.data.collate import PairedBatch, PairedBatchCollator
from prooflens.data.dataset import SourceImageDataset
from prooflens.data.sampling import FamilyBalancedTransformSampler, make_weighted_sampler
from prooflens.data.sampling import FixedTransformSampler
from prooflens.data.sampling import stable_seed
from prooflens.data.transforms import apply_transform
from prooflens.errors import TrainingError
from prooflens.inference.preprocess import create_dinov2_processor, preprocess_images
from prooflens.models.detector import DinoDetector
from prooflens.training.checkpoints import CheckpointManager
from prooflens.training.losses import SurvivalLossWeights, compute_survival_loss
from prooflens.training.hard_mining import HardTransformMiner
from prooflens.training.run_metadata import collect_run_metadata, config_hash, write_run_metadata


@dataclass(frozen=True, slots=True)
class TrainingResult:
    output_dir: Path
    best_checkpoint: Path
    best_composite_score: float


@dataclass(frozen=True, slots=True)
class ValidationSnapshot:
    clean_auc: float
    macro_robust_auc: float
    composite_score: float
    metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class TrainingComponents:
    model: nn.Module
    train_loader: DataLoader
    validation_loader: DataLoader
    collator: PairedBatchCollator
    validation_callback: Callable[[nn.Module, int], ValidationSnapshot] | None = None


def run_training(config: ExperimentConfig) -> TrainingResult:
    _seed_everything(config.seed)
    components = build_training_components(config)
    trainer = Trainer(
        config=config,
        model=components.model,
        train_loader=components.train_loader,
        validation_loader=components.validation_loader,
        collator=components.collator,
        validation_callback=components.validation_callback,
    )
    return trainer.fit()


def build_training_components(config: ExperimentConfig) -> TrainingComponents:
    manifest = pd.read_parquet(config.data.manifest)
    if "split" not in manifest or "split_group_id" not in manifest:
        raise TrainingError("training manifest must contain assigned split and split_group_id")
    train = manifest[manifest["split"] == "train"].reset_index(drop=True)
    validation = manifest[manifest["split"] == "validation"].reset_index(drop=True)
    if train.empty or validation.empty:
        raise TrainingError("training manifest must contain train and validation rows")
    processor = create_dinov2_processor()
    sampler = (
        FamilyBalancedTransformSampler()
        if config.transforms.enabled
        else FixedTransformSampler("jpeg_q90")
    )
    collator = PairedBatchCollator(processor=processor, sampler=sampler, seed=config.seed)
    train_sampler = make_weighted_sampler(train, seed=config.seed, num_samples=len(train))
    validation_collator = PairedBatchCollator(
        processor=processor, sampler=FixedTransformSampler("jpeg_q90"), seed=config.seed
    )
    train_loader = DataLoader(
        SourceImageDataset(train),
        batch_size=config.training.batch_size,
        sampler=train_sampler,
        num_workers=config.training.num_workers,
        collate_fn=collator,
    )
    validation_loader = DataLoader(
        SourceImageDataset(validation),
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        collate_fn=validation_collator,
    )
    model = DinoDetector.from_pretrained(config.model.name)
    model.set_trainable_stage(config.model.stage)
    return TrainingComponents(model, train_loader, validation_loader, collator)


class Trainer:
    def __init__(
        self,
        *,
        config: ExperimentConfig,
        model: nn.Module,
        train_loader: DataLoader,
        validation_loader: DataLoader | None = None,
        collator: PairedBatchCollator | None = None,
        validation_callback: Callable[[nn.Module, int], ValidationSnapshot] | None = None,
        checkpoints: CheckpointManager | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.collator = collator
        self.validation_callback = validation_callback
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = _resolve_device(device)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            [parameter for parameter in self.model.parameters() if parameter.requires_grad],
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
        )
        total_steps = max(1, math.ceil(len(train_loader) / config.training.gradient_accumulation_steps)) * config.training.epochs
        warmup_steps = int(total_steps * config.training.warmup_fraction)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: _cosine_warmup(step, warmup_steps, total_steps)
        )
        self.checkpoints = checkpoints or CheckpointManager(self.output_dir / "checkpoints")
        self.global_step = 0
        self.start_epoch = 1
        self.weights = SurvivalLossWeights(
            clean_bce=config.loss.clean_bce,
            transformed_bce=config.loss.transformed_bce,
            prediction_consistency=config.loss.prediction_consistency,
            feature_consistency=config.loss.feature_consistency,
        )
        self.current_epoch = 0
        self.hard_miner = (
            HardTransformMiner(
                seed=config.seed,
                candidate_count=config.transforms.candidate_count,
                exploration_probability=config.transforms.exploration_probability,
            )
            if config.transforms.hard_mining
            else None
        )
        self.selected_transform_counts: dict[str, int] = {}
        self.candidate_transform_counts: dict[str, int] = {}
        self.selected_transform_family_counts: dict[str, int] = {}

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.current_epoch = epoch
        if self.collator is not None:
            self.collator.set_epoch(epoch)
        running_loss = 0.0
        self.optimizer.zero_grad(set_to_none=True)
        accumulation = self.config.training.gradient_accumulation_steps
        for step, batch in enumerate(self.train_loader, start=1):
            try:
                raw_loss = self.compute_batch_loss(batch)
                (raw_loss / accumulation).backward()
            except RuntimeError as error:
                if "out of memory" in str(error).lower():
                    suggested = max(1, self.config.training.batch_size // 2)
                    raise TrainingError(
                        "CUDA out of memory at batch size "
                        f"{self.config.training.batch_size}; rerun with batch size <= {suggested}"
                    ) from error
                raise
            running_loss += float(raw_loss.detach().cpu())
            if step % accumulation == 0 or step == len(self.train_loader):
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.max_gradient_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1
        return running_loss / max(1, len(self.train_loader))

    def compute_batch_loss(self, batch: PairedBatch) -> Tensor:
        clean = batch.clean_pixels.to(self.device, non_blocking=True)
        transformed = batch.transformed_pixels.to(self.device, non_blocking=True)
        labels = batch.labels.to(self.device, non_blocking=True)
        if self.hard_miner is not None and batch.source_images is not None and self.collator is not None:
            transformed = self._hard_transform_pixels(batch).to(self.device, non_blocking=True)
        clean_output = self.model(clean)
        transformed_output = self.model(transformed)
        return compute_survival_loss(
            clean_output, transformed_output, labels, weights=self.weights
        ).total

    def _hard_transform_pixels(self, batch: PairedBatch) -> Tensor:
        assert self.hard_miner is not None
        candidate_specs = self.hard_miner.sample_candidates(batch.sample_ids, self.current_epoch)
        candidate_images = []
        candidate_ids = []
        for image, specs, sample_id in zip(
            batch.source_images or (), candidate_specs, batch.sample_ids, strict=True
        ):
            row_images = []
            row_ids = []
            for spec in specs:
                self.candidate_transform_counts[spec.family] = self.candidate_transform_counts.get(spec.family, 0) + 1
                row_images.append(
                    apply_transform(image, spec, stable_seed(self.config.seed, self.current_epoch, sample_id, spec.condition_id))
                )
                row_ids.append(spec.condition_id)
            candidate_images.extend(row_images)
            candidate_ids.append(tuple(row_ids))
        candidate_pixels = preprocess_images(
            candidate_images, processor=self.collator.processor if self.collator is not None else None
        ).to(self.device)
        was_training = self.model.training
        self.model.eval()
        with torch.no_grad():
            candidate_output = self.model(candidate_pixels)
        if was_training:
            self.model.train()
        candidate_logits = candidate_output.logits.reshape(len(batch.sample_ids), self.hard_miner.candidate_count)
        selected = self.hard_miner.select(
            candidate_logits.detach().cpu(), candidate_ids, batch.labels.cpu(),
            batch.sample_ids, self.current_epoch,
        )
        for condition_id in selected:
            self.selected_transform_counts[condition_id] = self.selected_transform_counts.get(condition_id, 0) + 1
            family = (
                "color_jitter" if condition_id.startswith("color_jitter") else
                "center_crop" if condition_id.startswith("center_crop") else
                condition_id.split("_", 1)[0]
            )
            self.selected_transform_family_counts[family] = self.selected_transform_family_counts.get(family, 0) + 1
        selected_images = [
            candidate_images[row * self.hard_miner.candidate_count + candidate_ids[row].index(selected[row])]
            for row in range(len(selected))
        ]
        return preprocess_images(
            selected_images, processor=self.collator.processor if self.collator is not None else None
        )

    def validate(self, epoch: int) -> ValidationSnapshot:
        if self.validation_callback is not None:
            return self.validation_callback(self.model, epoch)
        return _default_validation(self.model, self.validation_loader, self.device)

    def fit(self) -> TrainingResult:
        metadata_path = self.output_dir / "run_metadata.json"
        if not metadata_path.exists():
            write_run_metadata(collect_run_metadata(self.config), metadata_path)
        config_path = self.output_dir / "config.yaml"
        if not config_path.exists():
            config_path.write_text(
                yaml.safe_dump(self.config.model_dump(mode="json"), sort_keys=False),
                encoding="utf-8",
            )
        best_score = -math.inf
        best_checkpoint: Path | None = None
        epochs_without_improvement = 0
        history_path = self.output_dir / "history.jsonl"
        for epoch in range(self.start_epoch, self.config.training.epochs + 1):
            train_loss = self.train_epoch(epoch)
            validation = self.validate(epoch)
            checkpoint = self.checkpoints.save_epoch(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                global_step=self.global_step,
                config_hash=config_hash(self.config),
            )
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "clean_auc": validation.clean_auc,
                "macro_robust_auc": validation.macro_robust_auc,
                "composite_score": validation.composite_score,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
                "hard_mining_selected": dict(sorted(self.selected_transform_counts.items())),
                "hard_mining_candidate_family_proportions": self._family_proportions(self.candidate_transform_counts),
                "hard_mining_selected_family_proportions": self._family_proportions(self.selected_transform_family_counts),
            }
            with history_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, sort_keys=True) + "\n")
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
        return TrainingResult(self.output_dir, best_checkpoint, float(best_score))

    def _family_proportions(self, counts: dict[str, int]) -> dict[str, float]:
        total = sum(counts.values())
        proportions = {
            family: count / total for family, count in sorted(counts.items())
        } if total else {}
        overloaded = [family for family, value in proportions.items() if value > 0.60]
        if overloaded:
            warnings.warn(
                "hard-transform selection family exceeds 60 percent: "
                + ", ".join(overloaded),
                RuntimeWarning,
                stacklevel=2,
            )
        return proportions


def _default_validation(model, loader, device) -> ValidationSnapshot:
    if loader is None:
        return ValidationSnapshot(0.5, 0.5, 0.5, {})
    from sklearn.metrics import roc_auc_score

    labels: list[float] = []
    clean_scores: list[float] = []
    transformed_scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            clean_output = model(batch.clean_pixels.to(device))
            transformed_output = model(batch.transformed_pixels.to(device))
            labels.extend(batch.labels.tolist())
            clean_scores.extend(torch.sigmoid(clean_output.logits).cpu().tolist())
            transformed_scores.extend(torch.sigmoid(transformed_output.logits).cpu().tolist())
    clean_auc = _safe_auc(labels, clean_scores)
    robust_auc = _safe_auc(labels, transformed_scores)
    return ValidationSnapshot(clean_auc, robust_auc, 0.5 * (clean_auc + robust_auc), {})


def _safe_auc(labels, scores) -> float:
    return float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(labels, scores)) if len(set(labels)) == 2 else 0.5


def _cosine_warmup(step: int, warmup_steps: int, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(1e-8, step / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
