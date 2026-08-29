from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from uuid import uuid4

from prooflens.config import ExperimentConfig
from prooflens.data.hashing import sha256_file
from prooflens.errors import DataIntegrityError


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


def config_sha256(config: ExperimentConfig) -> str:
    payload = config.model_dump_json().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def collect_run_metadata(config: ExperimentConfig, *, device: str) -> RunMetadata:
    split_path = Path(config.data.manifest)
    split_hash = sha256_file(split_path)
    provenance_path = split_path.with_suffix(".json")
    if not provenance_path.is_file():
        raise DataIntegrityError(f"split metadata is missing: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    recorded_split = str(provenance.get("split_sha256", ""))
    source_hash = str(provenance.get("source_manifest_sha256", ""))
    if recorded_split != split_hash or len(source_hash) != 64:
        raise DataIntegrityError("split metadata hashes do not match the assigned manifest")
    packages = ("torch", "torchvision", "transformers", "datasets", "onnxruntime")
    return RunMetadata(
        git_commit=_git_commit(),
        python_version=platform.python_version(),
        package_versions={name: _package_version(name) for name in packages},
        operating_system=platform.platform(),
        device=device,
        manifest_sha256=source_hash,
        split_sha256=split_hash,
        config_sha256=config_sha256(config),
        seed=config.seed,
        started_at_utc=datetime.now(UTC).isoformat(),
        backbone=config.model.name,
    )


def write_run_metadata(value: RunMetadata, path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(asdict(value), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not-installed"
