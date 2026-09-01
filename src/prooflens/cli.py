"""Unified cross-platform command-line entry point for ProofLens."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path

import yaml
from pydantic import ValidationError

from prooflens.config import ExperimentConfig, load_config
from prooflens.errors import (
    DataIntegrityError,
    ExportError,
    ProofLensError,
    TrainingError,
    UserInputError,
)

COMMANDS = (
    "acquire",
    "manifest",
    "audit",
    "split",
    "train",
    "evaluate",
    "evaluate-stress",
    "select",
    "calibrate",
    "report",
    "export",
    "predict",
    "app",
)

# Kept as a public registry so integrations can inspect the supported command
# surface without invoking any optional runtime dependencies.
COMMAND_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="prooflens", description="Robust AI image detector workflow"
    )
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
    split.add_argument("--minimum-holdout-family-rows", type=int, default=100)
    train = subparsers.choices["train"]
    source = train.add_mutually_exclusive_group(required=True)
    source.add_argument("--config", type=Path)
    source.add_argument("--config-from-selection", type=Path)
    train.add_argument("--seed", type=int)
    train.add_argument("--output", type=Path)
    train.add_argument("--resume-from", type=Path)
    evaluate = subparsers.choices["evaluate"]
    source = evaluate.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path)
    source.add_argument("--selection", type=Path)
    evaluate.add_argument("--suite", choices=("clean", "clean-robust-generator"), required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate_stress = subparsers.choices["evaluate-stress"]
    evaluate_stress.add_argument("--selection", type=Path, required=True)
    evaluate_stress.add_argument("--split", choices=("validation", "test"), default="validation")
    evaluate_stress.add_argument("--output", type=Path, required=True)
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
    predict = subparsers.choices["predict"]
    predict.add_argument("--backend", choices=("torch", "onnx"), default="onnx")
    predict.add_argument("--checkpoint", type=Path)
    predict.add_argument("--model", type=Path)
    predict.add_argument("--calibration", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument(
        "--preprocessing",
        choices=("auto", "dinov2", "fixture"),
        default="auto",
        help="Auto-detect from artifact_manifest.json; explicit modes support legacy bundles.",
    )
    predict.add_argument("--artifact-manifest", type=Path)
    app = subparsers.choices["app"]
    app.add_argument("--backend", choices=("torch", "onnx"), default="onnx")
    app.add_argument("--checkpoint", type=Path)
    app.add_argument("--model", type=Path)
    app.add_argument("--calibration", type=Path, required=True)
    app.add_argument(
        "--preprocessing",
        choices=("auto", "dinov2", "fixture"),
        default="auto",
        help="Auto-detect from artifact_manifest.json; explicit modes support legacy bundles.",
    )
    app.add_argument("--artifact-manifest", type=Path)
    return parser


def dispatch(args: argparse.Namespace) -> int:
    handlers = COMMAND_HANDLERS or {
        "acquire": run_acquire_cli,
        "manifest": run_manifest_cli,
        "audit": run_audit_cli,
        "split": run_split_cli,
        "train": run_train_cli,
        "evaluate": run_evaluate_cli,
        "evaluate-stress": run_evaluate_stress_cli,
        "select": run_select_cli,
        "calibrate": run_calibrate_cli,
        "report": run_report_cli,
        "export": run_export_cli,
        "predict": run_predict_cli,
        "app": run_app_cli,
    }
    try:
        return int(handlers[args.command](args))
    except UserInputError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2
    except (OSError, json.JSONDecodeError, KeyError) as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2
    except (yaml.YAMLError, ValidationError) as error:
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
    except Exception as error:
        if _is_arrow_exception(error):
            print(f"data integrity error: {error}", file=sys.stderr)
            return 3
        raise


def main() -> None:
    raise SystemExit(dispatch(build_parser().parse_args()))


def _is_arrow_exception(error: BaseException) -> bool:
    """Identify optional Parquet errors without importing PyArrow at CLI startup."""

    try:
        import pyarrow as pa
    except ImportError:
        return False
    return isinstance(error, pa.ArrowException)


def run_acquire_cli(args: argparse.Namespace) -> int:
    from prooflens.data.acquire import acquire_sid_subset

    result = acquire_sid_subset(args.config, args.output)
    print(result.manifest_path)
    return 0


def run_manifest_cli(args: argparse.Namespace) -> int:
    from prooflens.data.acquire import load_primary_policy, validate_primary_manifest
    from prooflens.data.adapters.aigenimages2026 import AIGenImages2026Adapter
    from prooflens.data.adapters.local_manifest import CanonicalParquetAdapter
    from prooflens.data.adapters.wildfake import WildFakeAdapter
    from prooflens.data.manifest import build_manifest

    policy = load_primary_policy(args.config)
    adapters = []
    for source in policy.sources:
        if source.name == "sid_set":
            adapters.append(CanonicalParquetAdapter(source.root / "manifest.parquet", "sid_set"))
        elif source.name == "aigenimages2026":
            adapters.append(
                AIGenImages2026Adapter(
                    root=source.root,
                    version="073e1924d9d0d85ac97a53b07947b6ac95ce241c",
                )
            )
        elif source.name == "wildfake":
            adapters.append(
                WildFakeAdapter(
                    root=source.root,
                    version="configured",
                )
            )
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

    policy = SplitPolicy(
        args.seed,
        0.10,
        0.10,
        frozenset(),
        frozenset(),
        minimum_holdout_family_rows=args.minimum_holdout_family_rows,
    )
    result = write_split_manifest(
        pd.read_parquet(args.manifest), args.output, policy, args.manifest
    )
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
    import yaml

    from prooflens.training.trainer import run_training

    config = _config_from_train_args(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "config.yaml").write_text(
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    result = run_training(config, resume_from=getattr(args, "resume_from", None))
    print(result.best_checkpoint)
    return 0


def run_evaluate_cli(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    import pandas as pd
    import torch
    from torch.utils.data import DataLoader

    from prooflens.data.collate import PairedBatchCollator
    from prooflens.data.dataset import SourceImageDataset
    from prooflens.data.sampling import FixedTransformSampler
    from prooflens.data.transforms import canonical_specs
    from prooflens.evaluation.metrics import compute_condition_auc, compute_metrics
    from prooflens.evaluation.predict import predict_loader
    from prooflens.inference.preprocess import create_dinov2_processor
    from prooflens.inference.torch_backend import TorchLogitBackend
    from prooflens.models.detector import DinoDetector

    run_dir = _resolve_run_dir(args.run, args.selection)
    config = load_config(run_dir / "config.yaml").resolve(Path.cwd())
    checkpoint = run_dir / "checkpoints" / "best.pt"
    processor = create_dinov2_processor()
    backend = TorchLogitBackend.from_checkpoint(
        checkpoint,
        model_factory=lambda: DinoDetector.from_pretrained(config.model.name),
        processor=processor,
    )
    model = backend.model
    evaluation_device = "cuda" if torch.cuda.is_available() else "cpu"
    frame = pd.read_parquet(config.data.manifest)
    evaluation_split = args.split
    selected = frame[frame["split"] == evaluation_split].reset_index(drop=True)
    specs = canonical_specs()
    records: list[pd.DataFrame] = []
    for index, spec in enumerate(specs):
        collator = PairedBatchCollator(
            processor=processor,
            sampler=FixedTransformSampler(spec.condition_id),
            seed=config.seed,
        )
        loader = DataLoader(
            SourceImageDataset(selected),
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )
        predicted = predict_loader(
            model,
            loader,
            checkpoint_id=checkpoint.stem,
            device=evaluation_device,
            condition_override=spec.condition_id,
        )
        if index == 0:
            records.append(predicted[predicted["condition_id"] == "clean"])
        if args.suite == "clean-robust-generator":
            records.append(predicted[predicted["condition_id"] == spec.condition_id])
        if args.suite == "clean":
            break

    generator_split = "generator_validation" if args.split == "validation" else "generator_test"
    if args.suite == "clean-robust-generator":
        generator_rows = frame[frame["split"] == generator_split].reset_index(drop=True)
        collator = PairedBatchCollator(
            processor=processor,
            sampler=FixedTransformSampler(specs[0].condition_id),
            seed=config.seed,
        )
        loader = DataLoader(
            SourceImageDataset(generator_rows),
            batch_size=config.training.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )
        generator = predict_loader(
            model,
            loader,
            checkpoint_id=checkpoint.stem,
            device=evaluation_device,
        )
        records.append(generator[generator["condition_id"] == "clean"])

    predictions = pd.concat(records, ignore_index=True)
    prediction_path = run_dir / f"predictions-{args.split}.parquet"
    predictions.to_parquet(prediction_path, index=False)
    (run_dir / "report").mkdir(parents=True, exist_ok=True)
    if args.suite == "clean":
        payload = {"clean_auc": compute_condition_auc(predictions)}
        metrics_name = f"metrics-{args.split}-clean.json"
    else:
        report = compute_metrics(
            predictions,
            evaluation_split=args.split,
            generator_split=generator_split,
        )
        payload = {"ranking": asdict(report)}
        metrics_name = "metrics.json" if args.split == "validation" else "metrics-test.json"
    (run_dir / "report" / metrics_name).write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(prediction_path)
    return 0


def run_evaluate_stress_cli(args: argparse.Namespace) -> int:
    """Evaluate the selected checkpoint on supplemental redistribution conditions."""

    import pandas as pd
    import torch

    from prooflens.data.dataset import SourceImageDataset
    from prooflens.evaluation.stress import (
        compute_stress_metrics,
        evaluate_stress,
        write_stress_predictions,
    )
    from prooflens.inference.preprocess import create_dinov2_processor
    from prooflens.inference.torch_backend import TorchLogitBackend
    from prooflens.models.detector import DinoDetector

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    run_dir = _resolve_run_dir(None, args.selection)
    _verified_validation_split_hash(run_dir, selection)
    config = load_config(run_dir / "config.yaml").resolve(Path.cwd())
    backend = TorchLogitBackend.from_checkpoint(
        run_dir / "checkpoints" / "best.pt",
        model_factory=lambda: DinoDetector.from_pretrained(config.model.name),
        processor=create_dinov2_processor(),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    selected = pd.read_parquet(config.data.manifest)
    selected = selected.loc[selected["split"] == args.split].reset_index(drop=True)
    predictions = evaluate_stress(
        SourceImageDataset(selected), backend, checkpoint_id="best", seed=config.seed
    )
    args.output.mkdir(parents=True, exist_ok=True)
    prediction_path = write_stress_predictions(predictions, args.output / "predictions-stress.parquet")
    (args.output / "stress-metrics.json").write_text(
        json.dumps(compute_stress_metrics(predictions), indent=2) + "\n", encoding="utf-8"
    )
    print(prediction_path)
    return 0


def run_select_cli(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    from prooflens.evaluation.select import Candidate, select_best

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
        candidates.append(
            Candidate(
                run_id,
                ranking["clean_auc"],
                ranking["macro_robust_auc"],
                ranking["worst_family_auc"],
                ranking.get("unseen_generator_auc", 0.0),
                ranking.get("model_parameters", 0),
            )
        )
    selected = select_best(candidates)
    selected_run = next(
        path
        for path in args.runs
        if (path.parent.name if path.name == "run" else path.name) == selected.checkpoint_id
    )
    metadata_path = selected_run / "run_metadata.json"
    if not metadata_path.is_file():
        raise UserInputError(f"selected run has no run_metadata.json artifact: {selected_run}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    split_hash = metadata.get("split_sha256")
    if not isinstance(split_hash, str) or len(split_hash) != 64:
        raise UserInputError("selected run metadata has no valid split_sha256")
    output = {
        "checkpoint_id": selected.checkpoint_id,
        "run_dir": str(selected_run),
        "config": str(selected_run / "config.yaml"),
        "validation_split_hash": split_hash,
        "candidate": asdict(selected),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=str) + "\n", encoding="utf-8")
    print(args.output)
    return 0


def run_calibrate_cli(args: argparse.Namespace) -> int:
    import pandas as pd
    import torch

    from prooflens.evaluation.calibration import (
        fit_temperature,
        select_operating_threshold,
        write_calibration,
    )

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    run_dir = _resolve_run_dir(
        Path(selection.get("run_dir", "")) if selection.get("run_dir") else None, args.selection
    )
    split_hash = _verified_validation_split_hash(run_dir, selection)
    prediction_path = run_dir / "predictions-validation.parquet"
    frame = pd.read_parquet(prediction_path)
    clean = frame[(frame.split == args.split) & (frame.condition_id == "clean")]
    scaler = fit_temperature(
        torch.tensor(clean.logit.to_numpy(), dtype=torch.float32),
        torch.tensor(clean.label.to_numpy(), dtype=torch.float32),
    )
    scores = (
        torch.sigmoid(scaler(torch.tensor(clean.logit.to_numpy(), dtype=torch.float32)))
        .detach()
        .numpy()
    )
    threshold = select_operating_threshold(scores, clean.label.to_numpy())
    write_calibration(
        temperature=float(scaler.temperature.item()),
        threshold=threshold,
        validation_split_hash=split_hash,
        path=args.output,
    )
    selection["calibration_path"] = str(args.output)
    args.selection.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    return 0


def run_report_cli(args: argparse.Namespace) -> int:
    import pandas as pd
    import torch

    from prooflens.evaluation.calibration import compute_threshold_metrics
    from prooflens.evaluation.metrics import compute_metrics
    from prooflens.reporting.plots import write_auc_plot
    from prooflens.reporting.gallery import select_error_cases, write_error_case_artifacts
    from prooflens.reporting.tables import write_metric_artifacts

    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    run_dir = _resolve_run_dir(
        Path(selection.get("run_dir", "")) if selection.get("run_dir") else None, args.selection
    )
    prediction_path = run_dir / "predictions-test.parquet"
    frame = pd.read_parquet(prediction_path)
    report = compute_metrics(frame, evaluation_split="test", generator_split="generator_test")
    calibration = _load_selected_calibration(selection)
    temperature = float(calibration["temperature"])
    threshold = float(calibration["threshold"])
    clean = frame.loc[(frame["split"] == "test") & (frame["condition_id"] == "clean")]
    calibrated_scores = torch.sigmoid(
        torch.tensor(clean["logit"].to_numpy(), dtype=torch.float64) / temperature
    ).numpy()
    operating = compute_threshold_metrics(
        calibrated_scores,
        clean["label"].to_numpy(),
        threshold,
    )
    write_metric_artifacts(report, operating, args.output)
    write_auc_plot(report, args.output / "auc.png")
    calibrated_frame = frame.copy()
    calibrated_frame["score"] = torch.sigmoid(
        torch.tensor(frame["logit"].to_numpy(), dtype=torch.float64) / temperature
    ).numpy()
    error_frame = calibrated_frame.loc[calibrated_frame["split"] == "test"]
    cases = select_error_cases(error_frame, threshold=threshold)
    write_error_case_artifacts(cases, args.output)
    print(args.output)
    return 0


def run_export_cli(args: argparse.Namespace) -> int:
    import pandas as pd
    import torch

    from prooflens.data.dataset import SourceImageDataset
    from prooflens.errors import ExportError
    from prooflens.export.onnx_export import export_onnx, verify_onnx_parity
    from prooflens.inference.artifacts import (
        ARTIFACT_MANIFEST_NAME,
        write_artifact_manifest,
    )
    from prooflens.inference.preprocess import (
        PREPROCESSING_VERSION,
        create_dinov2_processor,
        preprocess_images,
    )
    from prooflens.inference.torch_backend import TorchLogitBackend
    from prooflens.models.detector import DinoDetector

    if args.verify != 32:
        raise UserInputError("--verify must be 32 for the publication parity gate")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    run_dir = _resolve_run_dir(
        Path(selection.get("run_dir", "")) if selection.get("run_dir") else None, args.selection
    )
    config = load_config(run_dir / "config.yaml").resolve(Path.cwd())
    processor = create_dinov2_processor()
    backend = TorchLogitBackend.from_checkpoint(
        run_dir / "checkpoints" / "best.pt",
        model_factory=lambda: DinoDetector.from_pretrained(config.model.name),
        processor=processor,
    )
    frame = pd.read_parquet(config.data.manifest)
    parity_frame = frame.loc[frame["split"].isin(("validation", "test"))].head(32)
    if len(parity_frame) != 32:
        raise ExportError("ONNX parity requires 32 validation or test images")
    parity_items = SourceImageDataset(parity_frame.reset_index(drop=True))
    sample = preprocess_images(
        tuple(parity_items[index].image for index in range(len(parity_items))),
        processor=processor,
    )
    if not isinstance(sample, torch.Tensor):
        raise ExportError("preprocessing did not return a Torch tensor")
    calibration = _load_selected_calibration(selection)
    temperature = float(calibration["temperature"])
    path = _publish_verified_onnx(
        backend.model,
        sample,
        args.output,
        temperature=temperature,
        export_fn=export_onnx,
        verify_fn=verify_onnx_parity,
    )
    if args.format == "openvino":
        from prooflens.export.openvino_export import try_openvino_smoke

        try_openvino_smoke(
            path,
            sample[:1].detach().cpu().numpy(),
            path.with_name("openvino_report.json"),
        )
    calibration_path = Path(str(selection["calibration_path"]))
    published_files = {
        "calibration": calibration_path,
        "export_report": path.with_name("export_report.json"),
        "model": path,
        "selection": args.selection,
    }
    openvino_report = path.with_name("openvino_report.json")
    if openvino_report.is_file():
        published_files["openvino_report"] = openvino_report
    checkpoint_id = str(selection.get("checkpoint_id", "selected"))
    write_artifact_manifest(
        path.with_name(ARTIFACT_MANIFEST_NAME),
        artifact_tier=str(selection.get("artifact_tier", "production-candidate")),
        model_version=f"prooflens-{checkpoint_id}-onnx",
        preprocessing_name="dinov2",
        preprocessing_version=PREPROCESSING_VERSION,
        files=published_files,
    )
    print(path)
    return 0


def _verified_validation_split_hash(run_dir: Path, selection: dict[str, object]) -> str:
    from prooflens.data.hashing import sha256_file

    selected_hash = selection.get("validation_split_hash")
    if (
        not isinstance(selected_hash, str)
        or re.fullmatch(r"[0-9a-fA-F]{64}", selected_hash) is None
    ):
        raise UserInputError("selection has no valid validation_split_hash")
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise UserInputError(f"selected run has no run_metadata.json artifact: {run_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("split_sha256") != selected_hash:
        raise UserInputError("selection split hash does not match selected run metadata")
    config = load_config(run_dir / "config.yaml").resolve(Path.cwd())
    split_path = Path(config.data.manifest)
    sidecar_path = split_path.with_suffix(".json")
    if not split_path.is_file() or not sidecar_path.is_file():
        raise UserInputError("selected run split manifest or metadata is missing")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("split_sha256") != selected_hash or sha256_file(split_path) != selected_hash:
        raise UserInputError("selected run split artifact does not match calibration provenance")
    return selected_hash


def _load_selected_calibration(selection: dict[str, object]) -> dict[str, object]:
    calibration_value = selection.get("calibration_path")
    if not isinstance(calibration_value, str) or not calibration_value.strip():
        raise UserInputError("selection has no calibration_path; run calibrate first")
    calibration = json.loads(Path(calibration_value).read_text(encoding="utf-8"))
    selected_hash = selection.get("validation_split_hash")
    if calibration.get("validation_split_hash") != selected_hash:
        raise UserInputError("calibration split hash does not match the selected run")
    return calibration


def _publish_verified_onnx(
    model,
    sample,
    output: Path,
    *,
    temperature: float,
    export_fn,
    verify_fn,
) -> Path:
    from uuid import uuid4

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    staged_model = destination.with_name(
        f".{destination.stem}.unverified-{token}{destination.suffix}"
    )
    staged_report = destination.with_name(f".export-report-{token}.json")
    report_temporary = destination.with_name(f".export-report-{token}.tmp")
    final_report = destination.with_name("export_report.json")
    try:
        exported = Path(export_fn(model, sample, staged_model))
        if exported != staged_model or not staged_model.is_file():
            raise ExportError("ONNX exporter did not produce the requested staging artifact")
        verify_fn(
            model,
            staged_model,
            sample,
            temperature=temperature,
            report_path=staged_report,
            release_model_before_onnx=True,
        )
        report_payload = json.loads(staged_report.read_text(encoding="utf-8"))
        report_payload["onnx_path"] = str(destination)
        report_temporary.write_text(
            json.dumps(report_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        staged_model.replace(destination)
        report_temporary.replace(final_report)
        return destination
    finally:
        for temporary in (staged_model, staged_report, report_temporary):
            if temporary.exists():
                temporary.unlink()


def _build_inference_service(args: argparse.Namespace):
    from prooflens.inference.service import InferenceService

    if args.backend == "torch":
        if getattr(args, "preprocessing", "dinov2") == "fixture":
            raise UserInputError("fixture preprocessing is supported only by the ONNX demo")
        if args.checkpoint is None:
            raise UserInputError("--checkpoint is required for the torch backend")
        from prooflens.inference.preprocess import create_dinov2_processor
        from prooflens.inference.torch_backend import TorchLogitBackend
        from prooflens.models.detector import DinoDetector

        backend = TorchLogitBackend.from_checkpoint(
            args.checkpoint,
            model_factory=DinoDetector.from_pretrained,
            processor=create_dinov2_processor(),
        )
    else:
        if args.model is None:
            raise UserInputError("--model is required for the ONNX backend")
        from prooflens.inference.artifacts import (
            discover_artifact_manifest,
            load_artifact_metadata,
            validate_artifact_pair,
        )
        from prooflens.inference.onnx_backend import OnnxLogitBackend
        from prooflens.inference.preprocess import (
            FIXTURE_PREPROCESSING_VERSION,
            PREPROCESSING_VERSION,
            create_dinov2_processor,
            create_fixture_processor,
        )

        preprocessing = getattr(args, "preprocessing", "dinov2")
        manifest_path = getattr(args, "artifact_manifest", None)
        if manifest_path is None:
            manifest_path = discover_artifact_manifest(args.model)
        metadata = load_artifact_metadata(manifest_path) if manifest_path is not None else None
        if metadata is not None:
            validate_artifact_pair(
                metadata,
                model_path=args.model,
                calibration_path=args.calibration,
            )
        if preprocessing == "auto":
            if metadata is None:
                raise UserInputError(
                    "automatic preprocessing requires artifact_manifest.json; "
                    "use an explicit --preprocessing value only for a verified legacy bundle"
                )
            preprocessing = metadata.preprocessing_name
        elif metadata is not None and preprocessing != metadata.preprocessing_name:
            raise UserInputError(
                f"requested preprocessing {preprocessing!r} conflicts with artifact manifest "
                f"{metadata.preprocessing_name!r}"
            )
        expected_version = {
            "dinov2": PREPROCESSING_VERSION,
            "fixture": FIXTURE_PREPROCESSING_VERSION,
        }[preprocessing]
        if metadata is not None and metadata.preprocessing_version != expected_version:
            raise UserInputError(
                "artifact preprocessing version is unsupported: "
                f"{metadata.preprocessing_version}"
            )
        if preprocessing == "fixture":
            processor = create_fixture_processor()
        else:
            processor = create_dinov2_processor()
        model_version = metadata.model_version if metadata else "prooflens-onnx"

        backend = OnnxLogitBackend(
            args.model,
            processor,
            model_version=model_version,
            preprocessing_version=expected_version,
        )
    return InferenceService.from_calibration(backend, args.calibration)


def run_predict_cli(args: argparse.Namespace) -> int:
    from prooflens.inference.directory import write_directory_predictions

    output = write_directory_predictions(
        args.input,
        args.output,
        _build_inference_service(args),
    )
    print(output)
    return 0


def run_app_cli(args: argparse.Namespace) -> int:
    from prooflens.web.app import create_app

    app = create_app(_build_inference_service(args))
    app.launch(
        server_name="127.0.0.1",
        share=False,
        prevent_thread_lock=True,
    )
    block_thread = getattr(app, "block_thread", None)
    if callable(block_thread):
        block_thread()
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


COMMAND_HANDLERS.update(
    {
        "acquire": run_acquire_cli,
        "manifest": run_manifest_cli,
        "audit": run_audit_cli,
        "split": run_split_cli,
        "train": run_train_cli,
        "evaluate": run_evaluate_cli,
        "evaluate-stress": run_evaluate_stress_cli,
        "select": run_select_cli,
        "calibrate": run_calibrate_cli,
        "report": run_report_cli,
        "export": run_export_cli,
        "predict": run_predict_cli,
        "app": run_app_cli,
    }
)


if __name__ == "__main__":
    main()
