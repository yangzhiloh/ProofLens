"""Unified cross-platform command-line entry point for ProofLens."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

from prooflens.config import ExperimentConfig, load_config
from prooflens.errors import DataIntegrityError, ExportError, ProofLensError, TrainingError, UserInputError


COMMANDS = (
    "acquire", "manifest", "audit", "split", "train", "evaluate", "select",
    "calibrate", "report", "export", "app",
)

# Kept as a public registry so integrations can inspect the supported command
# surface without invoking any optional runtime dependencies.
COMMAND_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="prooflens", description="Robust AI image detector workflow")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        subparsers.add_parser(command)

    acquire = subparsers.choices["acquire"]
    acquire.add_argument("--config", type=Path, required=True)
    acquire.add_argument("--output", type=Path, required=True)
    manifest = subparsers.choices["manifest"]
    manifest.add_argument("--config", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)
    audit = subparsers.choices["audit"]
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    split = subparsers.choices["split"]
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=17)
    train = subparsers.choices["train"]
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--config-from-selection", type=Path)
    train.add_argument("--seed", type=int)
    train.add_argument("--output", type=Path)
    evaluate = subparsers.choices["evaluate"]
    source = evaluate.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path)
    source.add_argument("--selection", type=Path)
    evaluate.add_argument("--suite", choices=("clean", "clean-robust-generator"), required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    select = subparsers.choices["select"]
    select.add_argument("--runs", type=Path, nargs="+", required=True)
    select.add_argument("--output", type=Path, default=Path("artifacts/selection.json"))
    calibrate = subparsers.choices["calibrate"]
    calibrate.add_argument("--selection", type=Path, required=True)
    calibrate.add_argument("--split", choices=("validation",), default="validation")
    calibrate.add_argument("--output", type=Path, required=True)
    report = subparsers.choices["report"]
    report.add_argument("--selection", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    export = subparsers.choices["export"]
    export.add_argument("--selection", type=Path, required=True)
    export.add_argument("--format", choices=("onnx", "openvino"), default="onnx")
    export.add_argument("--verify", type=int, default=32)
    export.add_argument("--output", type=Path, required=True)
    app = subparsers.choices["app"]
    app.add_argument("--backend", choices=("torch", "onnx"), default="onnx")
    app.add_argument("--checkpoint", type=Path)
    app.add_argument("--model", type=Path)
    app.add_argument("--calibration", type=Path)
    return parser


def dispatch(args: argparse.Namespace) -> int:
    handlers = COMMAND_HANDLERS or {
        "acquire": run_acquire_cli, "manifest": run_manifest_cli, "audit": run_audit_cli,
        "split": run_split_cli, "train": run_train_cli, "evaluate": run_evaluate_cli,
        "select": run_select_cli, "calibrate": run_calibrate_cli, "report": run_report_cli,
        "export": run_export_cli, "app": run_app_cli,
    }
    try:
        return int(handlers[args.command](args))
    except UserInputError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2
    except DataIntegrityError as error:
        print(f"data integrity error: {error}", file=sys.stderr)
        return 3
    except (TrainingError, ExportError) as error:
        print(f"model error: {error}", file=sys.stderr)
        return 4
    except ProofLensError as error:
        print(f"prooflens error: {error}", file=sys.stderr)
        return 2


def main() -> None:
    raise SystemExit(dispatch(build_parser().parse_args()))


def run_acquire_cli(args: argparse.Namespace) -> int:
    from prooflens.data.acquire import acquire_sid_subset
    result = acquire_sid_subset(args.config, args.output)
    print(result.manifest_path)
    return 0


def run_manifest_cli(args: argparse.Namespace) -> int:
    import yaml
    from prooflens.data.adapters.sid_set import SidSetAdapter
    from prooflens.data.adapters.wildfake import WildFakeAdapter
    from prooflens.data.acquire import load_primary_policy, validate_primary_manifest
    from prooflens.data.manifest import build_manifest

    policy = load_primary_policy(args.config)
    adapters = []
    for source in policy.sources:
        if source.name == "sid_set":
            adapters.append(SidSetAdapter(version="configured", root=source.root))
        elif source.name == "wildfake":
            adapters.append(WildFakeAdapter(root=source.root, version="configured", generator_labeled=source.generator_labeled))
    result = build_manifest(adapters, args.output, policy.maximum_corrupt_fraction)
    frame = __import__("pandas").read_parquet(args.output)
    validate_primary_manifest(frame, policy)
    print(result.output_path)
    return 0


def run_audit_cli(args: argparse.Namespace) -> int:
    import pandas as pd
    from prooflens.data.audit import audit_manifest, write_audit
    paths = write_audit(audit_manifest(pd.read_parquet(args.manifest)), args.output)
    print(paths[0])
    return 0


def run_split_cli(args: argparse.Namespace) -> int:
    import pandas as pd
    from prooflens.data.splitting import SplitPolicy, write_split_manifest
    policy = SplitPolicy(args.seed, 0.10, 0.10, frozenset(), frozenset())
    result = write_split_manifest(pd.read_parquet(args.manifest), args.output, policy, args.manifest)
    print(result.split_sha256)
    return 0


def _config_from_train_args(args: argparse.Namespace) -> ExperimentConfig:
    if args.config is not None:
        config = load_config(args.config).resolve(Path.cwd())
    else:
        selection = json.loads(args.config_from_selection.read_text(encoding="utf-8"))
        config_path = Path(selection["config"])
        config = load_config(config_path).resolve(Path.cwd())
    raw = config.model_dump()
    if args.seed is not None:
        raw["seed"] = args.seed
    if args.output is not None:
        raw["output_dir"] = args.output
    return ExperimentConfig.model_validate(raw)


def run_train_cli(args: argparse.Namespace) -> int:
    from prooflens.training.trainer import run_training
    result = run_training(_config_from_train_args(args))
    print(result.best_checkpoint)
    return 0


def run_evaluate_cli(args: argparse.Namespace) -> int:
    from dataclasses import asdict
    import pandas as pd
    from torch.utils.data import DataLoader
    from prooflens.data.collate import PairedBatchCollator
    from prooflens.data.dataset import SourceImageDataset
    from prooflens.data.sampling import FixedTransformSampler
    from prooflens.data.transforms import canonical_specs
    from prooflens.evaluation.metrics import compute_metrics
    from prooflens.evaluation.predict import predict_loader
    from prooflens.inference.preprocess import create_dinov2_processor
    from prooflens.models.detector import DinoDetector
    from prooflens.training.checkpoints import CheckpointManager

    run_dir = _resolve_run_dir(args.run, args.selection)
    config = load_config(run_dir / "config.yaml").resolve(Path.cwd())
    checkpoint = run_dir / "checkpoints" / "best.pt"
    model = DinoDetector.from_pretrained(config.model.name)
    CheckpointManager(checkpoint.parent).load(checkpoint, model)
    frame = pd.read_parquet(config.data.manifest)
    processor = create_dinov2_processor()
    evaluation_split = args.split
    records = []
    for spec in canonical_specs():
        selected = frame[frame["split"] == evaluation_split].reset_index(drop=True)
        collator = PairedBatchCollator(processor=processor, sampler=FixedTransformSampler(spec.condition_id), seed=config.seed)
        loader = DataLoader(SourceImageDataset(selected), batch_size=config.training.batch_size, shuffle=False, collate_fn=collator, num_workers=0)
        records.append(predict_loader(model, loader, checkpoint_id=checkpoint.stem, condition_override=spec.condition_id))
    generator_split = "generator_validation" if args.split == "validation" else "generator_test"
    selected = frame[frame["split"] == generator_split].reset_index(drop=True)
    collator = PairedBatchCollator(processor=processor, sampler=FixedTransformSampler("jpeg_q90"), seed=config.seed)
    loader = DataLoader(SourceImageDataset(selected), batch_size=config.training.batch_size, shuffle=False, collate_fn=collator, num_workers=0)
    generator = predict_loader(model, loader, checkpoint_id=checkpoint.stem)
    records.append(generator[generator["condition_id"] == "clean"])
    predictions = pd.concat(records, ignore_index=True)
    prediction_path = run_dir / f"predictions-{args.split}.parquet"
    predictions.to_parquet(prediction_path, index=False)
    report = compute_metrics(predictions, evaluation_split=args.split, generator_split=generator_split)
    (run_dir / "report").mkdir(parents=True, exist_ok=True)
    (run_dir / "report" / f"metrics-{args.split}.json").write_text(json.dumps({"ranking": asdict(report)}, default=str) + "\n", encoding="utf-8")
    print(prediction_path)
    return 0


def run_select_cli(args: argparse.Namespace) -> int:
    from dataclasses import asdict
    from prooflens.evaluation.metrics import Candidate, select_best
    candidates = []
    for run in args.runs:
        metrics_path = run / "report" / "metrics.json"
        if not metrics_path.exists():
            metrics_path = run / "metrics.json"
        if not metrics_path.exists():
            metrics_path = run.parent / "report" / "metrics.json"
        if not metrics_path.exists():
            raise UserInputError(f"run has no metrics.json artifact: {run}")
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        ranking = payload.get("ranking", payload)
        run_id = run.parent.name if run.name == "run" else run.name
        candidates.append(Candidate(run_id, ranking["clean_auc"], ranking["macro_robust_auc"], ranking["worst_family_auc"], ranking.get("unseen_generator_auc", 0.0), ranking.get("model_parameters", 0)))
    selected = select_best(candidates)
    selected_run = next(path for path in args.runs if (path.parent.name if path.name == "run" else path.name) == selected.checkpoint_id)
    output = {"checkpoint_id": selected.checkpoint_id, "run_dir": str(selected_run), "config": str(selected_run / "config.yaml"), "candidate": asdict(selected)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=str) + "\n", encoding="utf-8")
    print(args.output)
    return 0


def run_calibrate_cli(args: argparse.Namespace) -> int:
    import numpy as np
    import pandas as pd
    import torch
    from prooflens.evaluation.calibration import compute_threshold_metrics, fit_temperature, select_operating_threshold, write_calibration
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    run_dir = _resolve_run_dir(Path(selection.get("run_dir", "")) if selection.get("run_dir") else None, args.selection)
    prediction_path = run_dir / "predictions-validation.parquet"
    frame = pd.read_parquet(prediction_path)
    clean = frame[(frame.split == args.split) & (frame.condition_id == "clean")]
    scaler = fit_temperature(torch.tensor(clean.logit.to_numpy(), dtype=torch.float32), torch.tensor(clean.label.to_numpy(), dtype=torch.float32))
    scores = torch.sigmoid(scaler(torch.tensor(clean.logit.to_numpy(), dtype=torch.float32))).detach().numpy()
    threshold = select_operating_threshold(scores, clean.label.to_numpy())
    operating = compute_threshold_metrics(scores, clean.label.to_numpy(), threshold)
    split_hash = selection.get("validation_split_hash", "unknown")
    write_calibration(scaler, operating, split_hash, args.output)
    print(args.output)
    return 0


def run_report_cli(args: argparse.Namespace) -> int:
    import pandas as pd
    from prooflens.evaluation.metrics import compute_metrics
    from prooflens.reporting.plots import write_auc_plot
    from prooflens.reporting.tables import write_metric_artifacts
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    run_dir = _resolve_run_dir(Path(selection.get("run_dir", "")) if selection.get("run_dir") else None, args.selection)
    prediction_path = run_dir / "predictions-validation.parquet"
    frame = pd.read_parquet(prediction_path)
    report = compute_metrics(frame)
    from prooflens.evaluation.calibration import ThresholdReport
    operating = ThresholdReport(0.5, 0.0, 0.0, 0.0, 0.0, 0, 0)
    write_metric_artifacts(report, operating, args.output)
    write_auc_plot(report, args.output / "auc.png")
    print(args.output)
    return 0


def run_export_cli(args: argparse.Namespace) -> int:
    from dataclasses import asdict
    import torch
    from prooflens.models.detector import DinoDetector
    from prooflens.training.checkpoints import CheckpointManager
    from prooflens.export.onnx_export import export_onnx, verify_onnx_parity
    if args.format == "openvino":
        from prooflens.export.openvino_export import smoke_openvino
        report = smoke_openvino(args.output.with_suffix(".onnx"), args.output.with_suffix(".json"))
        print(report.status)
        return 0
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    run_dir = _resolve_run_dir(Path(selection.get("run_dir", "")) if selection.get("run_dir") else None, args.selection)
    config = load_config(run_dir / "config.yaml").resolve(Path.cwd())
    model = DinoDetector.from_pretrained(config.model.name)
    CheckpointManager(run_dir / "checkpoints").load(run_dir / "checkpoints" / "best.pt", model)
    sample = torch.randn(max(1, min(args.verify, 32)), 3, 224, 224)
    path = export_onnx(model, sample, args.output)
    report = verify_onnx_parity(model, path, sample)
    path.with_name("export_report.json").write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    print(path)
    return 0


def run_app_cli(args: argparse.Namespace) -> int:
    from prooflens.inference.service import InferenceService
    from prooflens.web.app import create_app
    if args.backend == "torch":
        if args.checkpoint is None:
            raise UserInputError("--checkpoint is required for the torch backend")
        from prooflens.inference.torch_backend import TorchLogitBackend
        backend = TorchLogitBackend(args.checkpoint)
    else:
        if args.model is None:
            raise UserInputError("--model is required for the ONNX backend")
        from prooflens.inference.onnx_backend import OnnxLogitBackend
        from prooflens.inference.preprocess import create_dinov2_processor
        backend = OnnxLogitBackend(args.model, create_dinov2_processor(), "prooflens-onnx")
    app = create_app(InferenceService.from_calibration(backend, args.calibration))
    app.launch()
    return 0


def _resolve_run_dir(run: Path | None, selection: Path | None) -> Path:
    if run is not None:
        return Path(run)
    if selection is None:
        raise UserInputError("a run or selection is required")
    payload = json.loads(Path(selection).read_text(encoding="utf-8"))
    if payload.get("run_dir"):
        return Path(payload["run_dir"])
    checkpoint_id = payload.get("checkpoint_id")
    if checkpoint_id:
        return Path("artifacts/runs") / str(checkpoint_id)
    raise UserInputError("selection does not identify a run directory")


COMMAND_HANDLERS.update({
    "acquire": run_acquire_cli, "manifest": run_manifest_cli, "audit": run_audit_cli,
    "split": run_split_cli, "train": run_train_cli, "evaluate": run_evaluate_cli,
    "select": run_select_cli, "calibrate": run_calibrate_cli, "report": run_report_cli,
    "export": run_export_cli, "app": run_app_cli,
})


if __name__ == "__main__":
    main()
