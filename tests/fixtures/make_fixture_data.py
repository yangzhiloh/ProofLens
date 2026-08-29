"""Create the small, network-free fixture dataset used by integration tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from prooflens.data.hashing import perceptual_hash_file, sha256_file
from prooflens.data.schema import ManifestRecord, records_to_frame
from prooflens.data.splitting import SplitPolicy, write_split_manifest


def make_fixture_data(root: Path, per_class: int = 8, seed: int = 17) -> Path:
    if per_class < 8:
        raise ValueError("fixture data requires at least 8 images per class")
    root = Path(root)
    real_dir, fake_dir = root / "real", root / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for label, directory in ((0, real_dir), (1, fake_dir)):
        for index in range(per_class):
            image = _fixture_image(label, index, rng)
            image.save(directory / f"{'real' if label == 0 else 'fake'}-{index:03d}.png")
    return root


def build_fixture_manifest(root: Path, output_path: Path) -> pd.DataFrame:
    root = Path(root)
    records: list[ManifestRecord] = []
    for label, directory in ((0, root / "real"), (1, root / "fake")):
        for path in sorted(directory.glob("*.png")):
            sample_id = path.stem
            generator = (
                "authentic"
                if label == 0
                else f"fixture-generator-{int(sample_id.split('-')[-1]) % 3}"
            )
            with Image.open(path) as image:
                width, height = image.size
            records.append(
                ManifestRecord(
                    sample_id=sample_id,
                    path=path.resolve(),
                    label=label,
                    dataset_name="fixture",
                    dataset_version="1",
                    generator_family=generator,
                    source_group_id=sample_id,
                    original_image_id=sample_id,
                    width=width,
                    height=height,
                    file_format="PNG",
                    licence_identifier="MIT-fixture",
                    content_checksum=sha256_file(path),
                    perceptual_hash=perceptual_hash_file(path),
                )
            )
    frame = records_to_frame(records)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    return frame


def build_fixture_split(manifest: Path, output_path: Path, seed: int = 17) -> pd.DataFrame:
    manifest_path = Path(manifest)
    frame = pd.read_parquet(manifest_path)
    policy = SplitPolicy(
        seed=seed,
        validation_fraction=0.20,
        test_fraction=0.20,
        generator_validation_families=frozenset({"fixture-generator-1"}),
        generator_test_families=frozenset({"fixture-generator-2"}),
        generator_evaluation_real_fraction=0.10,
        max_phash_distance=4,
        minimum_holdout_family_rows=2,
    )
    result = write_split_manifest(frame, output_path, policy, manifest_path)
    return pd.read_parquet(result.output_path)


def _fixture_image(label: int, index: int, rng: np.random.Generator) -> Image.Image:
    size = 64
    base = 240 if label == 0 else 20
    noise = rng.integers(0, 12, size=(size, size, 3), dtype=np.uint8)
    pixels = np.clip(base + noise.astype(np.int16), 0, 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="RGB")
    draw = ImageDraw.Draw(image)
    for marker in range(8):
        x = int(rng.integers(2, size - 8))
        y = int(rng.integers(2, size - 8))
        color = tuple(int(value) for value in rng.integers(40, 220, size=3))
        draw.rectangle((x, y, x + 5, y + 5), fill=color)
    offset = int(rng.integers(0, 8))
    if label == 0:
        draw.rectangle((10 + offset, 12, 52 - offset, 52), outline=(30, 120, 220), width=4)
        draw.line((12, 54 - index % 8, 52, 10 + index % 8), fill=(220, 80, 40), width=3)
    else:
        draw.ellipse((10 + offset, 10, 54 - offset, 54), outline=(255, 220, 40), width=4)
        draw.line((10, 10, 54, 54), fill=(230, 50, 160), width=3)
        draw.line((54, 10, 10, 54), fill=(70, 220, 130), width=3)
    return image
