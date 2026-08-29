from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import pandas as pd

from prooflens.errors import MetricPartitionError
from prooflens.evaluation.calibration import ThresholdReport
from prooflens.evaluation.metrics import MetricReport

FAMILY_DISPLAY = {
    "jpeg": "JPEG",
    "blur": "Blur",
    "resize": "Resize",
    "noise": "Noise",
    "color_jitter": "Color jitter",
    "center_crop": "Center crop",
}


def metric_rows(report: MetricReport) -> list[tuple[str, float]]:
    """Flatten a metric report into the stable public report order."""

    if not isinstance(report, MetricReport):
        raise TypeError("report must be a MetricReport")
    missing = set(FAMILY_DISPLAY).difference(report.family_auc)
    if missing:
        raise MetricPartitionError(
            "metric report is missing canonical families: " + ", ".join(sorted(missing))
        )
    rows = [("Clean", report.clean_auc)]
    rows.extend(
        (FAMILY_DISPLAY[family], report.family_auc[family]) for family in FAMILY_DISPLAY
    )
    rows.extend(
        (f"Condition: {name}", value)
        for name, value in sorted(report.condition_auc.items())
    )
    rows.extend(
        [
            ("Macro robust", report.macro_robust_auc),
            ("Pooled robust", report.pooled_robust_auc),
            ("Worst family", report.worst_family_auc),
            ("Worst condition", report.worst_condition_auc),
            ("Unseen generator", report.unseen_generator_auc),
            ("Composite", report.composite_score),
        ]
    )
    for name, value in rows:
        if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise MetricPartitionError(f"report metric {name!r} must be a finite value in [0, 1]")
    return rows


def write_robustness_markdown(
    report: MetricReport,
    path: Path,
    threshold_report: ThresholdReport | None = None,
) -> Path:
    """Write the complete robustness and optional operating-point summary."""

    path = Path(path)
    lines = ["| Partition | ROC AUC |", "| --- | ---: |"]
    lines.extend(f"| {name} | {value:.6f} |" for name, value in metric_rows(report))
    lines.extend(
        [
            "",
            f"Model parameters: {report.model_parameters}",
            f"Median CPU inference time: {report.inference_ms_median:.3f} ms",
        ]
    )
    if threshold_report is not None:
        if not isinstance(threshold_report, ThresholdReport):
            raise TypeError("threshold_report must be a ThresholdReport or None")
        lines.extend(
            [
                f"Operating threshold: {threshold_report.threshold:.6f}",
                f"Accuracy: {threshold_report.accuracy:.6f}",
                f"Precision: {threshold_report.precision:.6f}",
                f"Recall: {threshold_report.recall:.6f}",
                f"F1: {threshold_report.f1:.6f}",
                f"False positives: {threshold_report.false_positives}",
                f"False negatives: {threshold_report.false_negatives}",
            ]
        )
    _atomic_write_text(path, "\n".join(lines) + "\n")
    return path


def write_metric_artifacts(
    report: MetricReport,
    threshold_report: ThresholdReport,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    """Write strict JSON, CSV, and Markdown views from the same typed reports."""

    rows = metric_rows(report)
    if not isinstance(threshold_report, ThresholdReport):
        raise TypeError("threshold_report must be a ThresholdReport")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "robustness.csv"
    markdown_path = output_dir / "robustness.md"
    payload = {
        "ranking": _json_safe(asdict(report)),
        "operating_point": asdict(threshold_report),
    }
    _atomic_write_text(
        json_path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    _atomic_write_csv(pd.DataFrame(rows, columns=["partition", "roc_auc"]), csv_path)
    write_robustness_markdown(report, markdown_path, threshold_report)
    return json_path, csv_path, markdown_path


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(text, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_csv(temporary_path, index=False)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
