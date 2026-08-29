"""Reproducibility metadata captured at the start of every training run."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from prooflens.config import ExperimentConfig
from prooflens.data.hashing import sha256_file


@dataclass(frozen=True, slots=True)
class RunMetadata:
    git_commit: str
    python_version: str
    package_versions: dict[str, str]
    operating_system: str
    device: str
    manifest_sha256: str
    split_sha256: str
    config_sha256: str
    seed: int
    started_at_utc: str
    backbone: str


def collect_run_metadata(config: ExperimentConfig) -> RunMetadata:
    manifest_hash = sha256_file(config.data.manifest)
    return RunMetadata(
        git_commit=_git_commit(),
        python_version=platform.python_version(),
        package_versions={name: _package_version(name) for name in (
            "torch", "torchvision", "transformers", "datasets", "onnxruntime"
        )},
        operating_system=platform.platform(),
        device="cuda" if _cuda_available() else "cpu",
        manifest_sha256=manifest_hash,
        split_sha256=manifest_hash,
        config_sha256=_config_hash(config),
        seed=config.seed,
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        backbone=config.model.name,
    )


def write_run_metadata(metadata: RunMetadata, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def config_hash(config: ExperimentConfig) -> str:
    return _config_hash(config)


def _config_hash(config: ExperimentConfig) -> str:
    import hashlib

    return hashlib.sha256(config.model_dump_json().encode("utf-8")).hexdigest()


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False
