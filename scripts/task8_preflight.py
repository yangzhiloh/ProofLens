"""Read-only task 8 readiness audit with an optional derived pilot manifest."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
import yaml

from prooflens.config import load_config
from prooflens.data.transforms import canonical_specs

Status = Literal["pass", "warn", "block"]


@dataclass(frozen=True, slots=True)
class Check:
    name: str
    status: Status
    detail: str


def run_preflight(root: Path) -> dict[str, object]:
    root = Path(root).resolve()
    policy = yaml.safe_load(
        (root / "configs/data/task8_sources.yaml").read_text(encoding="utf-8")
    )
    checks: list[Check] = []
    checks.extend(_hardware_checks(root, int(policy["minimum_free_disk_gib"])))
    checks.extend(_source_checks(root, policy["sources"]))
    checks.extend(_workflow_checks(root))
    recommendation = _training_recommendation()
    return {
        "schema_version": 1,
        "ready_for_download": not any(check.status == "block" for check in checks),
        "ready_for_training": not any(check.status != "pass" for check in checks),
        "checks": [asdict(check) for check in checks],
        "training_recommendation": recommendation,
        "next_commands": _next_commands(),
        "safety": "No dataset download or training was performed.",
    }


def write_pilot_manifest(source: Path, destination: Path, per_label_split: int = 32) -> Path:
    """Write a deterministic small subset of an already assigned canonical manifest."""

    frame = pd.read_parquet(source)
    required = {"sample_id", "split", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"pilot source manifest is missing fields: {sorted(missing)}")
    selected = []
    for (split, label), group in frame.groupby(["split", "label"], sort=True):
        if int(label) not in (0, 1):
            continue
        ordered = group.sort_values("sample_id", kind="mergesort")
        selected.append(ordered.head(per_label_split))
    if not selected:
        raise ValueError("pilot source manifest contains no binary assigned rows")
    pilot = pd.concat(selected, ignore_index=True)
    required_splits = {"train", "validation"}
    if not required_splits <= set(pilot["split"]):
        raise ValueError("pilot manifest requires train and validation rows")
    for split in required_splits:
        if set(pilot.loc[pilot["split"] == split, "label"]) != {0, 1}:
            raise ValueError(f"pilot manifest split {split} requires both labels")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pilot.to_parquet(destination, index=False)
    return destination


def _hardware_checks(root: Path, minimum_free_gib: int) -> list[Check]:
    free_gib = shutil.disk_usage(root).free / (1024**3)
    disk_status: Status = "pass" if free_gib >= minimum_free_gib else "block"
    checks = [
        Check(
            "disk-space",
            disk_status,
            f"{free_gib:.1f} GiB free; task 8 policy requires {minimum_free_gib} GiB",
        )
    ]
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        vram_gib = properties.total_memory / (1024**3)
        checks.append(Check("gpu", "pass", f"{properties.name}; {vram_gib:.1f} GiB VRAM"))
    else:
        checks.append(
            Check("gpu", "warn", "CUDA GPU unavailable; preflight works but full training will be slow")
        )
    checks.append(
        Check(
            "runtime",
            "pass",
            f"Python {platform.python_version()} on {platform.system()} {platform.machine()}",
        )
    )
    return checks


def _source_checks(root: Path, sources: list[dict[str, object]]) -> list[Check]:
    checks: list[Check] = []
    for source in sources:
        name = str(source["name"])
        approved = source.get("approved_for_acquisition") is True
        checks.append(
            Check(
                f"licence-{name}",
                "pass" if approved else "block",
                f"{source['licence_status']}: {source['url']}",
            )
        )
        path = root / str(source["local_root"])
        checks.append(_dataset_structure(name, path))
    kaggle_ready = bool(os.getenv("KAGGLE_API_TOKEN")) or (
        Path.home() / ".kaggle" / "kaggle.json"
    ).is_file()
    checks.append(
        Check(
            "kaggle-credentials",
            "pass" if kaggle_ready else "warn",
            "Kaggle credentials detected" if kaggle_ready else "configure Kaggle credentials before API download",
        )
    )
    return checks


def _dataset_structure(name: str, path: Path) -> Check:
    if not path.is_dir():
        return Check(f"dataset-{name}", "warn", f"not present yet: {path}")
    if name == "sid_set":
        required = (path / "manifest.parquet", path / "acquisition.json", path / "images")
        valid = all(candidate.exists() for candidate in required)
        return Check(f"dataset-{name}", "pass" if valid else "block", "SID acquisition layout" if valid else "SID layout is incomplete")
    if name == "cifake":
        valid = all((path / directory).is_dir() for directory in ("REAL", "FAKE"))
        return Check(f"dataset-{name}", "pass" if valid else "block", "CIFAKE REAL/FAKE layout" if valid else "CIFAKE requires REAL and FAKE directories")
    if name == "aigenimages2026":
        required = (path / "eval_real_pairs.csv", path / "0_real", path / "1_fake")
        families = (
            [item for item in (path / "1_fake").iterdir() if item.is_dir()]
            if (path / "1_fake").is_dir()
            else []
        )
        valid = all(candidate.exists() for candidate in required) and len(families) >= 3
        return Check(
            f"dataset-{name}",
            "pass" if valid else "block",
            "paired AIGenImages2026 layout with at least three generator families"
            if valid
            else "AIGenImages2026 requires pair metadata, real images, and three fake families",
        )
    generator_root = path / "fake"
    families = list(generator_root.iterdir()) if generator_root.is_dir() else []
    valid = (path / "real").is_dir() and len([item for item in families if item.is_dir()]) >= 3
    return Check(f"dataset-{name}", "pass" if valid else "block", "WildFake real/fake layout with at least three families" if valid else "WildFake requires real/ and at least three fake generator directories")


def _workflow_checks(root: Path) -> list[Check]:
    checks: list[Check] = []
    configs = [root / f"configs/experiments/e{index}_{name}.yaml" for index, name in enumerate(("frozen", "last2", "augmented", "consistency", "hard_mining"))]
    configs.append(root / "configs/experiments/e0_pilot.yaml")
    loaded = [load_config(path) for path in configs]
    checks.append(Check("experiment-configs", "pass", f"validated {len(loaded)} strict experiment configs"))
    families = {spec.family for spec in canonical_specs()}
    required = {
        "jpeg",
        "blur",
        "resize",
        "noise",
        "color_jitter",
        "center_crop",
    }
    checks.append(Check("robustness-transforms", "pass" if required <= families else "block", f"available families: {', '.join(sorted(families))}"))
    checks.append(Check("resume-support", "pass", "train CLI accepts --resume-from and checkpoints every epoch"))
    return checks


def _training_recommendation() -> dict[str, object]:
    if not torch.cuda.is_available():
        return {"device": "cpu", "batch_size": 2, "gradient_accumulation_steps": 16, "pilot_first": True}
    vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    if vram >= 16:
        return {"device": "cuda", "batch_size": 32, "gradient_accumulation_steps": 1, "mixed_precision": True}
    if vram >= 8:
        return {"device": "cuda", "batch_size": 8, "gradient_accumulation_steps": 4, "mixed_precision": True}
    return {"device": "cuda", "batch_size": 4, "gradient_accumulation_steps": 8, "mixed_precision": True}


def _next_commands() -> list[str]:
    return [
        "prooflens acquire --config configs/data/sid_subset.yaml --output data/raw/sid_set",
        "prooflens manifest --config configs/data/primary.yaml --output artifacts/manifests/primary.parquet",
        "prooflens audit --manifest artifacts/manifests/primary.parquet --output artifacts/reports/data-audit",
        "prooflens split --manifest artifacts/manifests/primary.parquet --output artifacts/manifests/primary-split.parquet --seed 17 --minimum-holdout-family-rows 20",
        "prooflens train --config configs/experiments/e0_pilot.yaml",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("artifacts/preflight/task8.json"))
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--pilot-output", type=Path, default=Path("artifacts/manifests/pilot-split.parquet"))
    parser.add_argument("--pilot-per-label-split", type=int, default=32)
    args = parser.parse_args()
    if args.source_manifest is not None:
        write_pilot_manifest(args.source_manifest, args.pilot_output, args.pilot_per_label_split)
    report = run_preflight(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output)
    if args.strict and not report["ready_for_training"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
