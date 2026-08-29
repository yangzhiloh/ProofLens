"""Run the deterministic miniature ProofLens workflow without downloads."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
import sys
import json
import time

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT / "src"))
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from prooflens.config import ExperimentConfig
from prooflens.data.collate import PairedBatchCollator
from prooflens.data.dataset import SourceImageDataset
from prooflens.data.sampling import FixedTransformSampler, make_weighted_sampler
from prooflens.data.transforms import canonical_specs
from prooflens.evaluation.metrics import compute_metrics
from prooflens.evaluation.predict import predict_loader, write_prediction_records
from prooflens.evaluation.calibration import compute_threshold_metrics, fit_temperature, select_operating_threshold
from prooflens.models.types import DetectorOutput
from prooflens.reporting.plots import write_auc_plot
from prooflens.reporting.tables import write_metric_artifacts
from prooflens.training.checkpoints import CheckpointManager
from prooflens.training.losses import SurvivalLossWeights
from prooflens.training.trainer import Trainer, TrainingResult
from tests.fixtures.make_fixture_data import build_fixture_manifest, build_fixture_split, make_fixture_data


class FixtureProcessor:
    def __call__(self, *, images, return_tensors: str):
        if return_tensors == "np":
            import numpy as np
            values = [np.asarray(image.resize((224, 224)), dtype=np.float32).transpose(2, 0, 1) / 255.0 for image in images]
            return {"pixel_values": np.stack(values).astype("float32")}
        values = [torch.from_numpy(__import__("numpy").asarray(image.resize((224, 224)), dtype="float32").transpose(2, 0, 1) / 255.0) for image in images]
        return {"pixel_values": torch.stack(values)}


class FixtureDetector(nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=5, stride=2), nn.ReLU(),
            nn.Conv2d(8, hidden_size, kernel_size=5, stride=2), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.feature_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)
        self.model_version = "prooflens-fixture"

    def forward(self, pixel_values: Tensor) -> DetectorOutput:
        features = self.features(pixel_values).flatten(1)
        normalized = self.feature_norm(features)
        logits = self.classifier(normalized).squeeze(-1)
        return DetectorOutput(logits=logits, features=nn.functional.normalize(normalized, dim=1))


@dataclass(frozen=True, slots=True)
class ReportResult:
    metrics_json: Path
    robustness_markdown: Path


@dataclass(frozen=True, slots=True)
class ReproductionResult:
    checkpoint: Path
    predictions: Path
    metrics: Path
    robustness_markdown: Path

    @classmethod
    def from_artifacts(cls, training: TrainingResult, predictions: Path, report: ReportResult) -> "ReproductionResult":
        return cls(training.best_checkpoint, predictions, report.metrics_json, report.robustness_markdown)


def reproduce_small(output_dir: Path) -> ReproductionResult:
    output_dir = Path(output_dir)
    fixture_root = make_fixture_data(output_dir / "fixture", per_class=8, seed=17)
    manifest = build_fixture_manifest(fixture_root, output_dir / "manifest.parquet")
    split = build_fixture_split(manifest, output_dir / "split.parquet", seed=17)
    started = time.perf_counter()
    training = train_fixture_model(split, output_dir / "run", seed=17, experiment="e3")
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions = evaluate_fixture_model(training.best_checkpoint, split, output_dir / "predictions.parquet")
    evaluation_seconds = time.perf_counter() - started
    inference_ms = benchmark_fixture_inference(training.best_checkpoint)
    report = build_fixture_report(predictions, output_dir / "report", inference_ms)
    _write_execution_metadata(output_dir, training_seconds, evaluation_seconds, inference_ms)
    return ReproductionResult.from_artifacts(training, predictions, report)


def run_fixture_experiment(output_dir: Path, experiment: str, seed: int = 17) -> ReproductionResult:
    output_dir = Path(output_dir)
    fixture_root = make_fixture_data(output_dir / "fixture", per_class=8, seed=seed)
    manifest = build_fixture_manifest(fixture_root, output_dir / "manifest.parquet")
    split = build_fixture_split(manifest, output_dir / "split.parquet", seed=seed)
    started = time.perf_counter()
    training = train_fixture_model(split, output_dir / "run", seed=seed, experiment=experiment)
    training_seconds = time.perf_counter() - started
    started = time.perf_counter()
    predictions = evaluate_fixture_model(training.best_checkpoint, split, output_dir / "predictions.parquet")
    evaluation_seconds = time.perf_counter() - started
    inference_ms = benchmark_fixture_inference(training.best_checkpoint)
    report = build_fixture_report(predictions, output_dir / "report", inference_ms)
    _write_execution_metadata(output_dir, training_seconds, evaluation_seconds, inference_ms)
    return ReproductionResult.from_artifacts(training, predictions, report)


def train_fixture_model(frame: pd.DataFrame, output_dir: Path, seed: int, experiment: str = "e3") -> TrainingResult:
    torch.manual_seed(seed)
    train_frame = frame[frame.split == "train"].reset_index(drop=True)
    val_frame = frame[frame.split == "validation"].reset_index(drop=True)
    processor = FixtureProcessor()
    enabled = experiment in {"e2", "e3", "e4"}
    sampler = FixedTransformSampler("jpeg_q90") if not enabled else FixedTransformSampler("noise_s0.10")
    collator = PairedBatchCollator(processor=processor, sampler=sampler, seed=seed)
    val_collator = PairedBatchCollator(processor=processor, sampler=FixedTransformSampler("jpeg_q90"), seed=seed)
    train_loader = DataLoader(
        SourceImageDataset(train_frame), batch_size=4,
        sampler=make_weighted_sampler(train_frame, seed=seed, num_samples=8),
        collate_fn=collator, num_workers=0,
    )
    val_loader = DataLoader(SourceImageDataset(val_frame), batch_size=4, shuffle=False, collate_fn=val_collator, num_workers=0)
    config = ExperimentConfig.model_validate({
        "seed": seed, "data": {"manifest": str(Path(output_dir).parent / "split.parquet")},
        "model": {"name": "prooflens-fixture", "stage": "head"},
        "training": {"epochs": 1, "batch_size": 4, "learning_rate": 0.01, "num_workers": 0},
        "transforms": {"enabled": enabled, "hard_mining": experiment == "e4", "candidate_count": 3, "exploration_probability": 0.20},
        "loss": {"clean_bce": 1.0, "transformed_bce": 1.0 if enabled else 0.0, "prediction_consistency": 0.25 if experiment in {"e3", "e4"} else 0.0, "feature_consistency": 0.10 if experiment in {"e3", "e4"} else 0.0},
        "output_dir": str(output_dir),
    })
    trainer = Trainer(config=config, model=FixtureDetector(), train_loader=train_loader, validation_loader=val_loader, collator=collator, checkpoints=CheckpointManager(output_dir / "checkpoints"))
    return trainer.fit()


def evaluate_fixture_model(checkpoint: Path, frame: pd.DataFrame, output_path: Path) -> Path:
    model = FixtureDetector()
    CheckpointManager(Path(checkpoint).parent).load(checkpoint, model)
    processor = FixtureProcessor()
    all_frames = []
    validation = frame[frame.split == "validation"].reset_index(drop=True)
    for spec in canonical_specs():
        collator = PairedBatchCollator(processor=processor, sampler=FixedTransformSampler(spec.condition_id), seed=17)
        loader = DataLoader(SourceImageDataset(validation), batch_size=4, shuffle=False, collate_fn=collator, num_workers=0)
        all_frames.append(predict_loader(model, loader, checkpoint_id=Path(checkpoint).stem, condition_override=spec.condition_id))
    unseen = frame[frame.split == "generator_validation"].reset_index(drop=True)
    collator = PairedBatchCollator(processor=processor, sampler=FixedTransformSampler("jpeg_q90"), seed=17)
    loader = DataLoader(SourceImageDataset(unseen), batch_size=4, shuffle=False, collate_fn=collator, num_workers=0)
    unseen_frame = predict_loader(model, loader, checkpoint_id=Path(checkpoint).stem)
    unseen_frame = unseen_frame[unseen_frame.condition_id == "clean"]
    result = pd.concat(all_frames + [unseen_frame], ignore_index=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    return output_path


def build_fixture_report(predictions: Path | pd.DataFrame, output_dir: Path, inference_ms: float | None = None) -> ReportResult:
    frame = pd.read_parquet(predictions) if isinstance(predictions, (str, Path)) else predictions
    report = compute_metrics(frame)
    report = replace(
        report,
        model_parameters=sum(parameter.numel() for parameter in FixtureDetector().parameters()),
        inference_ms_median=float("nan") if inference_ms is None else inference_ms,
    )
    clean = frame[(frame.split == "validation") & (frame.condition_id == "clean")]
    scaler = fit_temperature(torch.tensor(clean.logit.to_numpy(), dtype=torch.float32), torch.tensor(clean.label.to_numpy(), dtype=torch.float32))
    calibrated_scores = torch.sigmoid(scaler(torch.tensor(clean.logit.to_numpy(), dtype=torch.float32))).detach().numpy()
    threshold = select_operating_threshold(calibrated_scores, clean.label.to_numpy())
    threshold_report = compute_threshold_metrics(calibrated_scores, clean.label.to_numpy(), threshold)
    output_dir = Path(output_dir)
    metrics_json, _, markdown = write_metric_artifacts(report, threshold_report, output_dir)
    write_auc_plot(report, output_dir / "auc.png")
    return ReportResult(metrics_json, markdown)


def benchmark_fixture_inference(checkpoint: Path) -> float:
    model = FixtureDetector().eval()
    CheckpointManager(Path(checkpoint).parent).load(checkpoint, model)
    sample = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        model(sample)
        timings = []
        for _ in range(5):
            started = time.perf_counter()
            model(sample)
            timings.append((time.perf_counter() - started) * 1000.0)
    return float(__import__("numpy").median(timings))


def _write_execution_metadata(output_dir: Path, training_seconds: float, evaluation_seconds: float, inference_ms: float) -> None:
    (Path(output_dir) / "execution.json").write_text(
        json.dumps({
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "inference_ms_median": inference_ms,
        }, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke"))
    parser.add_argument("--experiment", choices=("e3", "e4"), default="e3")
    args = parser.parse_args()
    result = run_fixture_experiment(args.output, args.experiment)
    print(f"checkpoint={result.checkpoint}")
    print(f"predictions={result.predictions}")
    print(f"metrics={result.metrics}")
    print(f"robustness_markdown={result.robustness_markdown}")


if __name__ == "__main__":
    main()
