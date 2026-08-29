from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True, slots=True)
class CheckpointState:
    epoch: int
    global_step: int
    config_hash: str


class CheckpointManager:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        name: str,
        model: nn.Module,
        optimizer: Optimizer,
        epoch: int,
        global_step: int,
        config_hash: str,
        scheduler: object | None = None,
    ) -> Path:
        destination = self.directory / f"{name}.pt"
        temporary = self.directory / f".{name}.{uuid4().hex}.tmp"
        payload = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch,
            "global_step": global_step,
            "config_hash": config_hash,
            "python_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        try:
            torch.save(payload, temporary)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination

    def save_epoch(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: object,
        epoch: int,
        global_step: int,
        config_hash: str,
    ) -> Path:
        return self.save(
            f"epoch-{epoch}",
            model,
            optimizer,
            epoch,
            global_step,
            config_hash,
            scheduler,
        )

    def load(
        self,
        path: Path,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: object | None = None,
    ) -> CheckpointState:
        payload = torch.load(Path(path), map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        if scheduler is not None and payload["scheduler"] is not None:
            scheduler.load_state_dict(payload["scheduler"])
        random.setstate(payload["python_rng"])
        np.random.set_state(payload["numpy_rng"])
        torch.set_rng_state(payload["torch_rng"])
        if torch.cuda.is_available() and payload["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        return CheckpointState(
            epoch=int(payload["epoch"]),
            global_step=int(payload["global_step"]),
            config_hash=str(payload["config_hash"]),
        )

    def mark_best(self, checkpoint: Path) -> Path:
        source = Path(checkpoint)
        destination = self.directory / "best.pt"
        temporary = self.directory / f".best.{uuid4().hex}.tmp"
        try:
            shutil.copyfile(source, temporary)
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        return destination
