"""Atomic checkpoint persistence and resume state."""

from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from prooflens.errors import TrainingError


@dataclass(frozen=True, slots=True)
class RestoredCheckpoint:
    epoch: int
    global_step: int
    config_hash: str
    scheduler_state: dict[str, Any] | None = None


class CheckpointManager:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        epoch: int,
        global_step: int,
        config_hash: str,
        scheduler: Any | None = None,
    ) -> Path:
        if not name or Path(name).name != name:
            raise TrainingError("checkpoint name must be a simple nonempty filename")
        path = self.output_dir / (name if name.endswith(".pt") else f"{name}.pt")
        payload = _checkpoint_payload(model, optimizer, scheduler, epoch, global_step, config_hash)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            torch.save(payload, temporary)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def save_epoch(self, **kwargs: Any) -> Path:
        epoch = int(kwargs.pop("epoch"))
        return self.save(f"epoch-{epoch}", epoch=epoch, **kwargs)

    def mark_best(self, checkpoint: Path) -> Path:
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise TrainingError(f"cannot mark missing checkpoint as best: {checkpoint}")
        target = self.output_dir / "best.pt"
        temporary = target.with_suffix(".pt.tmp")
        try:
            shutil.copyfile(checkpoint, temporary)
            temporary.replace(target)
        finally:
            if temporary.exists():
                temporary.unlink()
        return target

    def load(
        self,
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> RestoredCheckpoint:
        path = Path(path)
        if not path.is_file():
            raise TrainingError(f"checkpoint does not exist: {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload["model"])
            if optimizer is not None:
                optimizer.load_state_dict(payload["optimizer"])
            if scheduler is not None and payload.get("scheduler") is not None:
                scheduler.load_state_dict(payload["scheduler"])
            _restore_rng(payload.get("rng"))
            return RestoredCheckpoint(
                epoch=int(payload["epoch"]),
                global_step=int(payload["global_step"]),
                config_hash=str(payload["config_hash"]),
                scheduler_state=payload.get("scheduler"),
            )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
            raise TrainingError(f"could not restore checkpoint {path}: {error}") from error


def _checkpoint_payload(model, optimizer, scheduler, epoch, global_step, config_hash):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config_hash": str(config_hash),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }


def _restore_rng(state):
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
