from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral

import torch
from torch import Tensor

from prooflens.data.dataset import SourceItem
from prooflens.data.sampling import TransformSampler, stable_seed
from prooflens.data.transforms import TransformSpec, apply_transform
from prooflens.errors import DataIntegrityError
from prooflens.inference.preprocess import ImageProcessor, preprocess_images


@dataclass(frozen=True, slots=True)
class PairedBatch:
    clean_pixels: Tensor
    transformed_pixels: Tensor
    labels: Tensor
    sample_ids: tuple[str, ...]
    condition_ids: tuple[str, ...]
    dataset_names: tuple[str, ...]
    generator_families: tuple[str, ...]
    source_group_ids: tuple[str, ...]
    splits: tuple[str, ...]
    split_group_ids: tuple[str, ...]


class PairedBatchCollator:
    def __init__(
        self,
        *,
        processor: ImageProcessor,
        sampler: TransformSampler,
        seed: int,
    ) -> None:
        if not callable(processor):
            raise DataIntegrityError("collator processor must be callable")
        if not callable(getattr(sampler, "sample", None)):
            raise DataIntegrityError("collator sampler must implement sample")
        self.processor = processor
        self.sampler = sampler
        self.seed = _nonnegative_integer(seed, "seed")
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = _nonnegative_integer(epoch, "epoch")

    def __call__(self, items: Sequence[SourceItem]) -> PairedBatch:
        batch = tuple(items)
        _validate_items(batch)
        specs = tuple(
            self.sampler.sample(item.sample_id, epoch=self.epoch, seed=self.seed)
            for item in batch
        )
        if any(not isinstance(spec, TransformSpec) for spec in specs):
            raise DataIntegrityError("transform sampler must return canonical TransformSpec values")
        transformed = tuple(
            apply_transform(
                item.image,
                spec,
                stable_seed(self.seed, self.epoch, item.sample_id, "transform-pixels"),
            )
            for item, spec in zip(batch, specs, strict=True)
        )
        clean_pixels = preprocess_images(
            [item.image.copy() for item in batch], processor=self.processor
        )
        transformed_pixels = preprocess_images(transformed, processor=self.processor)
        return PairedBatch(
            clean_pixels=clean_pixels,
            transformed_pixels=transformed_pixels,
            labels=torch.tensor(
                [item.label for item in batch], dtype=torch.float32
            ),
            sample_ids=tuple(item.sample_id for item in batch),
            condition_ids=tuple(spec.condition_id for spec in specs),
            dataset_names=tuple(item.dataset_name for item in batch),
            generator_families=tuple(item.generator_family for item in batch),
            source_group_ids=tuple(item.source_group_id for item in batch),
            splits=tuple(item.split for item in batch),
            split_group_ids=tuple(item.split_group_id for item in batch),
        )


def _validate_items(items: tuple[SourceItem, ...]) -> None:
    if not items:
        raise DataIntegrityError("paired collation requires a nonempty item sequence")
    if any(not isinstance(item, SourceItem) for item in items):
        raise DataIntegrityError("paired collation requires SourceItem values")
    identifiers = [item.sample_id for item in items]
    if any(not isinstance(value, str) or not value.strip() for value in identifiers):
        raise DataIntegrityError("paired collation sample_ids must be nonempty strings")
    if len(identifiers) != len(set(identifiers)):
        raise DataIntegrityError("paired collation sample_ids must be unique")
    if any(
        not isinstance(item.label, Integral)
        or isinstance(item.label, bool)
        or int(item.label) not in (0, 1)
        for item in items
    ):
        raise DataIntegrityError("paired collation labels must be binary 0 or 1")


def _nonnegative_integer(value: object, field: str) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool) or int(value) < 0:
        raise DataIntegrityError(f"collator {field} must be a nonnegative integer")
    return int(value)
