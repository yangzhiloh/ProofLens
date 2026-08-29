from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from prooflens.config import ExperimentConfig


class EpochAwareCollator:
    def __init__(self) -> None:
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


def _config(tmp_path: Path, *, epochs: int) -> ExperimentConfig:
    manifest = tmp_path / "assigned.parquet"
    manifest.write_bytes(b"fixture split")
    split_hash = hashlib.sha256(b"fixture split").hexdigest()
    manifest.with_suffix(".json").write_text(
        json.dumps(
            {
                "source_manifest_sha256": "a" * 64,
                "split_sha256": split_hash,
            }
        ),
        encoding="utf-8",
    )
    return ExperimentConfig.model_validate(
        {
            "seed": 17,
            "data": {"manifest": manifest},
            "model": {"name": "fixture/dino", "stage": "head"},
            "training": {
                "epochs": epochs,
                "batch_size": 2,
                "learning_rate": 0.01,
                "num_workers": 0,
            },
            "output_dir": tmp_path / "run",
        }
    )


def _components(config: ExperimentConfig):
    from prooflens.training.trainer import (
        TrainingComponents,
        ValidationSnapshot,
    )

    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    collator = EpochAwareCollator()

    def batch_loss(current_model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
        assert (config.output_dir / "run_metadata.json").exists()
        return current_model(batch).square().mean()

    return TrainingComponents(
        model=model,
        train_loader=[torch.ones(2, 2), torch.zeros(2, 2)],
        optimizer=optimizer,
        scheduler=scheduler,
        collator=collator,
        batch_loss_callback=batch_loss,
        validation_callback=lambda _model, _epoch: ValidationSnapshot(
            clean_auc=0.8,
            macro_robust_auc=0.6,
            composite_score=0.7,
            metrics={"fixture": 0.7},
        ),
        device="cpu",
        epoch_metrics_callback=lambda: {"selected/jpeg": 1.0},
    )


def test_tiny_training_emits_checkpoint_history_and_early_metadata(tmp_path) -> None:
    from prooflens.training.trainer import Trainer, run_training

    config = _config(tmp_path, epochs=1)
    components = _components(config)

    assert Trainer(config, components).amp_enabled is False

    result = run_training(config, components=components)
    history = [
        json.loads(line)
        for line in (result.output_dir / "history.jsonl").read_text().splitlines()
    ]

    assert result.best_checkpoint.exists()
    assert (result.output_dir / "run_metadata.json").exists()
    assert history[-1]["epoch"] == 1
    assert history[-1]["train_loss"] >= 0
    assert history[-1]["composite_score"] == 0.7
    assert history[-1]["training_metrics"] == {"selected/jpeg": 1.0}
    assert components.collator.epochs == [1]  # type: ignore[attr-defined]


def test_resume_continues_at_the_next_epoch(tmp_path) -> None:
    from prooflens.training.trainer import run_training

    first_config = _config(tmp_path, epochs=1)
    first = run_training(first_config, components=_components(first_config))
    resumed_config = _config(tmp_path, epochs=2)
    resumed_components = _components(resumed_config)

    result = run_training(
        resumed_config,
        components=resumed_components,
        resume_from=first.best_checkpoint,
    )
    history = [
        json.loads(line)
        for line in (result.output_dir / "history.jsonl").read_text().splitlines()
    ]

    assert [record["epoch"] for record in history] == [1, 2]
    assert resumed_components.collator.epochs == [2]  # type: ignore[attr-defined]
