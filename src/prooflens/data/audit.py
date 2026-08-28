from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from prooflens.errors import DatasetPolicyError

_QUANTILES = ((0.0, "min"), (0.25, "q25"), (0.5, "median"), (0.75, "q75"), (1.0, "max"))
_CATEGORICAL_FEATURES = ("dataset_name", "file_format", "generator_family")


@dataclass(frozen=True)
class AuditReport:
    row_count: int
    class_counts: dict[int, int]
    dataset_counts: dict[str, int]
    generator_counts: dict[str, int]
    dimension_quantiles: dict[str, dict[str, float | None]]
    file_format_crosstab: dict[str, dict[int, int]]
    missing_counts: dict[str, int]
    exact_duplicate_count: int
    perfect_shortcuts: tuple[str, ...]


def audit_manifest(frame: pd.DataFrame) -> AuditReport:
    """Summarize distributions and categorical label shortcuts in a manifest."""
    required = {"label", "dataset_name", "generator_family", "file_format", "width", "height"}
    missing = required - set(frame.columns)
    if missing:
        raise DatasetPolicyError(f"audit manifest is missing columns: {sorted(missing)}")

    class_counts = {
        int(key): int(value)
        for key, value in frame["label"].value_counts().sort_index().items()
    }
    dataset_counts = _string_counts(frame["dataset_name"])
    generator_counts = _string_counts(frame["generator_family"])
    dimension_quantiles = {
        column: _dimension_quantiles(frame[column]) for column in ("width", "height")
    }
    file_format_crosstab = _format_crosstab(frame)
    missing_counts = {
        column: _missing_count(frame[column]) for column in sorted(frame.columns)
    }

    if "content_checksum" in frame:
        checksums = _observed_strings(frame["content_checksum"])
        exact_duplicate_count = int(checksums.duplicated().sum())
    else:
        missing_counts["content_checksum"] = len(frame)
        exact_duplicate_count = 0

    perfect_shortcuts = tuple(
        column
        for column in _CATEGORICAL_FEATURES
        if _perfectly_predicts_label(frame, column)
    )
    return AuditReport(
        row_count=len(frame),
        class_counts=class_counts,
        dataset_counts=dataset_counts,
        generator_counts=generator_counts,
        dimension_quantiles=dimension_quantiles,
        file_format_crosstab=file_format_crosstab,
        missing_counts=missing_counts,
        exact_duplicate_count=exact_duplicate_count,
        perfect_shortcuts=perfect_shortcuts,
    )


def write_audit(report: AuditReport, output_dir: Path) -> tuple[Path, Path]:
    """Write machine-readable and human-readable audit artifacts atomically."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "audit.json"
    markdown_path = output_dir / "audit.md"
    json_text = json.dumps(asdict(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
    markdown_text = render_audit_markdown(report)
    _atomic_write_text(json_path, json_text)
    _atomic_write_text(markdown_path, markdown_text)
    return json_path, markdown_path


def render_audit_markdown(report: AuditReport) -> str:
    lines = ["# Manifest audit", "", f"## Row count\n\n{report.row_count}"]
    lines.extend(_mapping_section("Class counts", report.class_counts))
    lines.extend(_mapping_section("Dataset counts", report.dataset_counts))
    lines.extend(_mapping_section("Generator counts", report.generator_counts))
    lines.extend(["", "## Dimension quantiles", ""])
    for dimension, values in report.dimension_quantiles.items():
        rendered = ", ".join(f"{name}={value}" for name, value in values.items())
        lines.append(f"- {dimension}: {rendered}")
    lines.extend(["", "## File format by label", "", "| Format | Label 0 | Label 1 |", "| --- | ---: | ---: |"])
    lines.extend(
        f"| {name} | {counts.get(0, 0)} | {counts.get(1, 0)} |"
        for name, counts in report.file_format_crosstab.items()
    )
    lines.extend(_mapping_section("Missing metadata", report.missing_counts))
    lines.extend(["", "## Exact duplicates", "", str(report.exact_duplicate_count)])
    lines.extend(["", "## Perfect label shortcuts", ""])
    lines.extend(f"- {name}" for name in report.perfect_shortcuts)
    if not report.perfect_shortcuts:
        lines.append("None")
    return "\n".join(lines) + "\n"


def _dimension_quantiles(series: pd.Series) -> dict[str, float | None]:
    numeric = pd.to_numeric(series, errors="coerce")
    values = numeric.quantile([quantile for quantile, _ in _QUANTILES])
    return {
        name: None if pd.isna(values.loc[quantile]) else float(values.loc[quantile])
        for quantile, name in _QUANTILES
    }


def _format_crosstab(frame: pd.DataFrame) -> dict[str, dict[int, int]]:
    observed = frame.loc[
        ~_missing_mask(frame["file_format"]) & frame["label"].isin((0, 1)),
        ["file_format", "label"],
    ].copy()
    if observed.empty:
        return {}
    observed["file_format"] = observed["file_format"].astype(str).str.strip()
    table = pd.crosstab(observed["file_format"], observed["label"])
    return {
        str(name): {label: int(table.loc[name].get(label, 0)) for label in (0, 1)}
        for name in sorted(table.index.astype(str))
    }


def _perfectly_predicts_label(frame: pd.DataFrame, column: str) -> bool:
    if column not in frame or frame.empty:
        return False
    mask = ~_missing_mask(frame[column]) & frame["label"].isin((0, 1))
    observed = frame.loc[mask, [column, "label"]].copy()
    if len(observed) < 2:
        return False
    observed[column] = observed[column].astype(str).str.strip()
    if observed[column].nunique() < 2 or observed["label"].nunique() != 2:
        return False
    return bool(observed.groupby(column, dropna=False)["label"].nunique().max() == 1)


def _string_counts(series: pd.Series) -> dict[str, int]:
    counts = _observed_strings(series).value_counts()
    ordered = sorted(
        ((str(key), int(value)) for key, value in counts.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return dict(ordered)


def _observed_strings(series: pd.Series) -> pd.Series:
    return series.loc[~_missing_mask(series)].astype(str).str.strip()


def _missing_count(series: pd.Series) -> int:
    return int(_missing_mask(series).sum())


def _missing_mask(series: pd.Series) -> pd.Series:
    return series.isna() | series.fillna("").astype(str).str.strip().eq("")


def _mapping_section(title: str, values: dict[Any, Any]) -> list[str]:
    lines = ["", f"## {title}", ""]
    lines.extend(f"- {key}: {value}" for key, value in values.items())
    if not values:
        lines.append("None")
    return lines


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
