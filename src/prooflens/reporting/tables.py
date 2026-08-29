"""CSV, JSON, and Markdown robustness reports."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from prooflens.evaluation.calibration import ThresholdReport
from prooflens.evaluation.metrics import MetricReport


def metric_rows(report: MetricReport) -> list[tuple[str, float]]:
    rows = [("Clean", report.clean_auc)]
    display = {
        "jpeg": "JPEG", "blur": "Blur", "resize": "Resize", "noise": "Noise",
        "color_jitter": "Color jitter", "center_crop": "Center crop",
    }
    rows.extend((display.get(name, name), report.family_auc[name]) for name in display if name in report.family_auc)
    rows.extend((f"Condition: {name}", value) for name, value in sorted(report.condition_auc.items()))
    rows.extend([
        ("Macro robust", report.macro_robust_auc), ("Pooled robust", report.pooled_robust_auc),
        ("Worst family", report.worst_family_auc), ("Worst condition", report.worst_condition_auc),
        ("Unseen generator", report.unseen_generator_auc), ("Composite", report.composite_score),
    ])
    return rows


def write_robustness_markdown(
    report: MetricReport,
    path: Path,
    threshold_report: ThresholdReport | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| Partition | ROC AUC |", "| --- | ---: |"]
    lines.extend(f"| {name} | {value:.6f} |" for name, value in metric_rows(report))
    lines.extend([
        "", f"Model parameters: {report.model_parameters}",
        f"Median CPU inference time: {report.inference_ms_median:.3f} ms",
    ])
    if threshold_report is not None:
        lines.extend([
            f"Operating threshold: {threshold_report.threshold:.6f}",
            f"Accuracy: {threshold_report.accuracy:.6f}",
            f"Precision: {threshold_report.precision:.6f}",
            f"Recall: {threshold_report.recall:.6f}",
            f"F1: {threshold_report.f1:.6f}",
            f"False positives: {threshold_report.false_positives}",
            f"False negatives: {threshold_report.false_negatives}",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_metric_artifacts(
    report: MetricReport,
    threshold_report: ThresholdReport,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path, markdown_path = (
        output_dir / "metrics.json", output_dir / "robustness.csv", output_dir / "robustness.md"
    )
    payload = {"ranking": asdict(report), "operating_point": asdict(threshold_report)}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=True) + "\n", encoding="utf-8")
    pd.DataFrame(metric_rows(report), columns=["partition", "roc_auc"]).to_csv(csv_path, index=False)
    write_robustness_markdown(report, markdown_path, threshold_report)
    return json_path, csv_path, markdown_path
