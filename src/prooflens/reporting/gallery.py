from __future__ import annotations

import hashlib
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from prooflens.errors import DataIntegrityError, MetricPartitionError

_PREDICTION_COLUMNS = frozenset(
    {"sample_id", "label", "score", "condition_id", "generator_family"}
)
_MANIFEST_COLUMNS = frozenset({"sample_id", "path", "dataset_name"})


@dataclass(frozen=True, slots=True)
class ErrorCases:
    false_positives: pd.DataFrame
    false_negatives: pd.DataFrame
    unstable: pd.DataFrame


def select_error_cases(
    predictions: pd.DataFrame, per_category: int = 20, threshold: float = 0.5
) -> ErrorCases:
    """Select unique, confidence-ranked errors and transformation instabilities."""

    _validate_predictions(predictions, per_category, threshold)
    enriched = _attach_clean_scores(predictions.copy())
    identity = [
        column
        for column in ("checkpoint_id", "split", "sample_id", "condition_id")
        if column in enriched.columns
    ]
    unique = enriched.drop_duplicates(identity, keep="first")
    clean_unique = unique.loc[unique["condition_id"] == "clean"]
    false_positives = (
        clean_unique.loc[
            (clean_unique["label"] == 0) & (clean_unique["score"] >= threshold)
        ]
        .sort_values("score", ascending=False, kind="mergesort")
        .head(per_category)
        .reset_index(drop=True)
    )
    false_negatives = (
        clean_unique.loc[
            (clean_unique["label"] == 1) & (clean_unique["score"] < threshold)
        ]
        .sort_values("score", ascending=True, kind="mergesort")
        .head(per_category)
        .reset_index(drop=True)
    )
    unstable = (
        unique.loc[unique["condition_id"] != "clean"]
        .dropna(subset=["absolute_change"])
        .sort_values("absolute_change", ascending=False, kind="mergesort")
        .head(per_category)
        .reset_index(drop=True)
    )
    return ErrorCases(false_positives, false_negatives, unstable)


def write_error_case_artifacts(
    cases: ErrorCases, output_dir: Path
) -> tuple[Path, Path]:
    """Write licensing-safe, sample-level JSON and Markdown error summaries."""

    if not isinstance(cases, ErrorCases):
        raise TypeError("cases must be an ErrorCases instance")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    categories = {
        "false_positives": _error_records(cases.false_positives),
        "false_negatives": _error_records(cases.false_negatives),
        "unstable": _error_records(cases.unstable),
    }
    json_path = output_dir / "error-cases.json"
    markdown_path = output_dir / "error-cases.md"
    _atomic_write_text(
        json_path,
        json.dumps(categories, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    lines = ["# Representative error cases", ""]
    for title, key in (
        ("False positives", "false_positives"),
        ("False negatives", "false_negatives"),
        ("Most unstable", "unstable"),
    ):
        records = categories[key]
        lines.extend([f"## {title}", ""])
        if not records:
            lines.extend(["No cases selected.", ""])
            continue
        columns = list(records[0])
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join("---" for _ in columns) + " |")
        for record in records:
            lines.append(
                "| "
                + " | ".join(_markdown_value(record[column]) for column in columns)
                + " |"
            )
        lines.append("")
    _atomic_write_text(markdown_path, "\n".join(lines))
    return json_path, markdown_path


def write_error_gallery(cases: ErrorCases, manifest: pd.DataFrame, output_dir: Path) -> Path:
    """Write escaped HTML and bounded thumbnails without copying source images."""

    if not isinstance(cases, ErrorCases):
        raise TypeError("cases must be an ErrorCases instance")
    _validate_manifest(manifest)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    thumbnail_dir = output_dir / "thumbnails"
    thumbnail_dir.mkdir(exist_ok=True)
    metadata = manifest.set_index("sample_id")
    sections: list[str] = []
    for title, frame in (
        ("False positives", cases.false_positives),
        ("False negatives", cases.false_negatives),
        ("Most unstable", cases.unstable),
    ):
        cards: list[str] = []
        for row in frame.itertuples(index=False):
            if row.sample_id not in metadata.index:
                raise DataIntegrityError(
                    f"gallery sample {row.sample_id!r} is missing from the manifest"
                )
            source_record = metadata.loc[row.sample_id]
            source = Path(source_record["path"])
            thumb_name = (
                hashlib.sha256(str(row.sample_id).encode("utf-8")).hexdigest()[:16]
                + ".jpg"
            )
            thumb_path = thumbnail_dir / thumb_name
            _write_thumbnail(source, thumb_path)
            change = getattr(row, "absolute_change", float("nan"))
            change_text = f"{change:.4f}" if math.isfinite(float(change)) else "n/a"
            caption = html.escape(
                f"{row.sample_id} | dataset={source_record['dataset_name']} | path={source} | "
                f"label={row.label} | score={row.score:.4f} | condition={row.condition_id} | "
                f"generator={row.generator_family} | clean_to_transformed_change={change_text}"
            )
            cards.append(
                '<figure><img src="thumbnails/'
                + thumb_name
                + '" alt="sample thumbnail"><figcaption>'
                + caption
                + "</figcaption></figure>"
            )
        sections.append(f"<h2>{html.escape(title)}</h2>" + "".join(cards))
    document = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>ProofLens errors</title></head><body>"
        + "".join(sections)
        + "</body></html>"
    )
    gallery_path = output_dir / "error-gallery.html"
    _atomic_write_text(gallery_path, document)
    return gallery_path


def _attach_clean_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    join_keys = [
        column
        for column in ("checkpoint_id", "split", "sample_id")
        if column in predictions.columns
    ]
    if "sample_id" not in join_keys:
        join_keys.append("sample_id")
    clean = predictions.loc[predictions["condition_id"] == "clean", [*join_keys, "score"]]
    if clean.duplicated(join_keys).any():
        raise MetricPartitionError("clean predictions must be unique for stability matching")
    clean = clean.rename(columns={"score": "clean_score"})
    enriched = predictions.merge(clean, on=join_keys, how="left", validate="many_to_one")
    enriched["absolute_change"] = (enriched["score"] - enriched["clean_score"]).abs()
    return enriched


def _error_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    preferred = (
        "sample_id",
        "label",
        "score",
        "condition_id",
        "generator_family",
        "clean_score",
        "absolute_change",
        "split",
        "checkpoint_id",
    )
    columns = [column for column in preferred if column in frame.columns]
    records: list[dict[str, object]] = []
    for row in frame.loc[:, columns].itertuples(index=False, name=None):
        record: dict[str, object] = {}
        for column, value in zip(columns, row, strict=True):
            if pd.isna(value):
                record[column] = None
            elif isinstance(value, np.generic):
                record[column] = value.item()
            else:
                record[column] = value
        records.append(record)
    return records


def _markdown_value(value: object) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        text = f"{value:.6f}"
    else:
        text = str(value)
    escaped = text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")
    return html.escape(escaped, quote=False)


def _validate_predictions(
    predictions: pd.DataFrame, per_category: int, threshold: float
) -> None:
    if not isinstance(predictions, pd.DataFrame):
        raise MetricPartitionError("predictions must be a pandas DataFrame")
    missing = _PREDICTION_COLUMNS.difference(predictions.columns)
    if missing:
        raise MetricPartitionError(
            "predictions are missing required gallery columns: " + ", ".join(sorted(missing))
        )
    if not isinstance(per_category, int) or isinstance(per_category, bool) or per_category <= 0:
        raise MetricPartitionError("per_category must be a positive integer")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
        or not 0.0 <= float(threshold) <= 1.0
    ):
        raise MetricPartitionError("gallery threshold must be a finite probability")
    labels = pd.to_numeric(predictions["label"], errors="coerce").to_numpy(dtype=float)
    scores = pd.to_numeric(predictions["score"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(labels).all() or not np.isfinite(scores).all():
        raise MetricPartitionError("gallery labels and scores must be finite")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise MetricPartitionError("gallery labels must be binary")
    if not np.logical_and(scores >= 0.0, scores <= 1.0).all():
        raise MetricPartitionError("gallery scores must be probabilities in [0, 1]")
    for column in ("sample_id", "condition_id", "generator_family"):
        values = predictions[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise MetricPartitionError(f"gallery column {column!r} must contain nonempty strings")


def _validate_manifest(manifest: pd.DataFrame) -> None:
    if not isinstance(manifest, pd.DataFrame):
        raise DataIntegrityError("gallery manifest must be a pandas DataFrame")
    missing = _MANIFEST_COLUMNS.difference(manifest.columns)
    if missing:
        raise DataIntegrityError(
            "gallery manifest is missing required columns: " + ", ".join(sorted(missing))
        )
    if manifest["sample_id"].duplicated().any():
        raise DataIntegrityError("gallery manifest contains duplicate sample IDs")
    if manifest["sample_id"].isna().any() or manifest["sample_id"].astype(str).str.strip().eq("").any():
        raise DataIntegrityError("gallery manifest sample IDs must be nonempty")


def _write_thumbnail(source: Path, destination: Path) -> None:
    with Image.open(source) as image:
        thumbnail = ImageOps.fit(ImageOps.exif_transpose(image).convert("RGB"), (224, 224))
        thumbnail.save(destination, format="JPEG", quality=85)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(text, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
