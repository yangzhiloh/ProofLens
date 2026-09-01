"""Acquire a balanced SID subset resumably, one pinned Parquet shard at a time."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

from prooflens.data.acquire import (
    ACQUISITION_METADATA_NAME,
    SidAcquisitionConfig,
    hash_acquisition_config,
)
from prooflens.data.adapters.sid_set import SidSetAdapter
from prooflens.data.licences import SID_SET
from prooflens.data.manifest import build_manifest
from prooflens.errors import DatasetAcquisitionError, UserInputError


class _RecordsAdapter:
    def __init__(self, records: list[object]) -> None:
        self.records = records

    def scan(self):
        yield from self.records


def _load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"completed_shards": [], "counts": {"0": 0, "1": 0}, "rows": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path, state: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _decode_image(value: object, image_id: str) -> Image.Image:
    if not isinstance(value, dict):
        raise DatasetAcquisitionError(f"SID row {image_id} has no encoded image mapping")
    encoded = value.get("bytes")
    source_path = value.get("path")
    try:
        if isinstance(encoded, bytes):
            with Image.open(io.BytesIO(encoded)) as source:
                return source.convert("RGB")
        if isinstance(source_path, str) and source_path:
            with Image.open(source_path) as source:
                return source.convert("RGB")
    except (OSError, ValueError) as error:
        raise DatasetAcquisitionError(f"SID row {image_id} contains an invalid image") from error
    raise DatasetAcquisitionError(f"SID row {image_id} has no usable encoded image")


def _save_row(row: dict[str, object], partial: Path) -> dict[str, object]:
    image_id = str(row["img_id"])
    label = int(row["label"])
    relative = Path("images") / str(label) / (
        hashlib.sha256(image_id.encode("utf-8")).hexdigest() + ".png"
    )
    destination = partial / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = _decode_image(row["image"], image_id)
    image.save(destination, format="PNG")
    return {
        "img_id": image_id,
        "label": label,
        "image_path": str(destination),
        "width": image.width,
        "height": image.height,
        "file_format": "PNG",
    }


def acquire(config_path: Path, output: Path) -> Path:
    config = SidAcquisitionConfig.from_value(config_path)
    output = output.resolve()
    if output.exists():
        raise UserInputError(f"acquisition output root already exists: {output}")
    partial = output.with_name(output.name + ".partial")
    partial.mkdir(parents=True, exist_ok=True)
    state_path = partial / "shard-progress.json"
    state = _load_state(state_path)
    counts = {int(key): int(value) for key, value in dict(state["counts"]).items()}
    completed = set(str(value) for value in state["completed_shards"])
    saved_rows = list(state["rows"])
    saved_ids = {str(row["img_id"]) for row in saved_rows}

    files = HfApi().list_repo_files(
        config.dataset_id, repo_type="dataset", revision=config.revision
    )
    shards = sorted(
        name
        for name in files
        if name.startswith(f"data/{config.split}-") and name.endswith(".parquet")
    )
    if not shards:
        raise DatasetAcquisitionError(
            f"no {config.split} Parquet shards found at pinned revision {config.revision}"
        )

    print(f"Found {len(shards)} pinned {config.split} shards", flush=True)
    for index, shard in enumerate(shards, start=1):
        if counts[0] >= config.per_class and counts[1] >= config.per_class:
            break
        if shard in completed:
            print(f"[{index}/{len(shards)}] already complete: {shard}", flush=True)
            continue
        print(
            f"[{index}/{len(shards)}] downloading {shard}; counts={counts}",
            flush=True,
        )
        local_path = hf_hub_download(
            repo_id=config.dataset_id,
            filename=shard,
            repo_type="dataset",
            revision=config.revision,
        )
        parquet = pq.ParquetFile(local_path)
        for batch in parquet.iter_batches(
            batch_size=32,
            columns=[config.id_field, config.label_field, config.image_field],
        ):
            for row in batch.to_pylist():
                label = int(row[config.label_field])
                if label not in (0, 1) or counts[label] >= config.per_class:
                    continue
                image_id = str(row[config.id_field])
                if image_id in saved_ids:
                    continue
                saved = _save_row(row, partial)
                saved_rows.append(saved)
                saved_ids.add(image_id)
                counts[label] += 1
        completed.add(shard)
        state = {
            "completed_shards": sorted(completed),
            "counts": {str(key): value for key, value in counts.items()},
            "rows": saved_rows,
        }
        _write_state(state_path, state)
        print(f"[{index}/{len(shards)}] complete; counts={counts}", flush=True)

    if counts != {0: config.per_class, 1: config.per_class}:
        raise DatasetAcquisitionError(
            f"SID shards exhausted before balanced target: {counts}; expected {config.per_class} each"
        )

    records = list(SidSetAdapter(version=config.revision).scan_rows(saved_rows))
    build_manifest(
        [_RecordsAdapter(records)],
        partial / config.manifest_name,
        max_corrupt_fraction=0.0,
    )
    metadata = {
        "config_sha256": hash_acquisition_config(config),
        "counts": counts,
        "dataset_id": config.dataset_id,
        "dataset_revision": config.revision,
        "licence_identifier": SID_SET.licence_identifier,
        "observed_dataset_revision": config.revision,
        "split": config.split,
        "acquisition_method": "resumable-pinned-parquet-shards",
        "completed_shards": sorted(completed),
    }
    (partial / ACQUISITION_METADATA_NAME).write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    state_path.unlink(missing_ok=True)
    partial.replace(output)

    relocated = [
        record.model_copy(update={"path": output / record.path.relative_to(partial)})
        for record in records
    ]
    build_manifest(
        [_RecordsAdapter(relocated)],
        output / config.manifest_name,
        max_corrupt_fraction=0.0,
    )
    print(output / config.manifest_name, flush=True)
    return output / config.manifest_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    acquire(args.config, args.output)


if __name__ == "__main__":
    main()
