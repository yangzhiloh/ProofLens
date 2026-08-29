"""Safe, thumbnail-only error galleries."""

from __future__ import annotations

import hashlib
import html
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image


@dataclass(frozen=True, slots=True)
class ErrorCases:
    false_positives: pd.DataFrame
    false_negatives: pd.DataFrame
    unstable: pd.DataFrame


def select_error_cases(predictions: pd.DataFrame, per_category: int = 20) -> ErrorCases:
    if per_category < 1:
        raise ValueError("per_category must be positive")
    false_positives = predictions[predictions["label"] == 0].nlargest(per_category, "score")
    false_negatives = predictions[predictions["label"] == 1].nsmallest(per_category, "score")
    clean_rows = predictions[predictions["condition_id"] == "clean"]
    clean = clean_rows[["sample_id", "score"]].rename(columns={"score": "clean_score"})
    changed = predictions[predictions["condition_id"] != "clean"].merge(clean, on="sample_id")
    changed = changed.assign(absolute_change=(changed["score"] - changed["clean_score"]).abs())
    unstable = changed.nlargest(per_category, "absolute_change")
    return ErrorCases(false_positives, false_negatives, unstable)


def write_error_gallery(cases: ErrorCases, manifest: pd.DataFrame, output_dir: Path) -> Path:
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
        for row in frame.itertuples():
            if row.sample_id not in metadata.index:
                continue
            source_record = metadata.loc[row.sample_id]
            source = Path(source_record.path)
            thumb_name = hashlib.sha256(str(row.sample_id).encode()).hexdigest()[:16] + ".jpg"
            thumb_path = thumbnail_dir / thumb_name
            try:
                with Image.open(source) as image:
                    image.convert("RGB").resize((224, 224)).save(thumb_path, quality=85)
            except (OSError, ValueError):
                continue
            caption = html.escape(
                f"{row.sample_id} | dataset={source_record.dataset_name} | path={source} | "
                f"label={row.label} | score={row.score:.4f} | condition={row.condition_id} | "
                f"generator={source_record.generator_family} | "
                f"absolute_change={getattr(row, 'absolute_change', float('nan')):.4f}"
            )
            cards.append(
                f'<figure><img src="thumbnails/{thumb_name}" alt="sample">'
                f"<figcaption>{caption}</figcaption></figure>"
            )
        sections.append(f"<h2>{html.escape(title)}</h2>" + "".join(cards))
    path = output_dir / "error-gallery.html"
    path.write_text(
        "<!doctype html><meta charset='utf-8'><title>ProofLens errors</title>" + "".join(sections),
        encoding="utf-8",
    )
    return path
