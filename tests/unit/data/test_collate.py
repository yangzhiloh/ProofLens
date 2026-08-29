from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from prooflens.errors import DataIntegrityError


def _task6_types():
    from prooflens.data.collate import PairedBatchCollator
    from prooflens.data.dataset import SourceItem
    from prooflens.data.sampling import FixedTransformSampler

    return PairedBatchCollator, SourceItem, FixedTransformSampler


def _detail_image(offset: int) -> Image.Image:
    y, x = np.indices((18, 22))
    pixels = np.stack(
        (
            (x * 17 + y * 5 + offset) % 256,
            (x * 3 + y * 19 + offset * 2) % 256,
            (x * 11 + y * 7 + offset * 3) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _source_items(count: int = 2):
    _, SourceItem, _ = _task6_types()
    return tuple(
        SourceItem(
            image=_detail_image(index + 1),
            label=index % 2,
            sample_id=f"sample-{index}",
            dataset_name="sid_set" if index % 2 == 0 else "wildfake",
            generator_family="authentic" if index % 2 == 0 else "sdxl",
            source_group_id=f"source-{index}",
            split="train",
            split_group_id=f"split-{index}",
        )
        for index in range(count)
    )


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, ...]] = []

    def __call__(self, *, images, return_tensors: str):
        if return_tensors != "pt":
            raise AssertionError("processor tensor format changed")
        captured = tuple(np.asarray(image).copy() for image in images)
        self.calls.append(captured)
        tensors = []
        for image in images:
            resized = image.resize((224, 224), Image.Resampling.BICUBIC)
            array = np.asarray(resized, dtype=np.float32).copy() / 255.0
            tensors.append(torch.from_numpy(array).permute(2, 0, 1))
        return {"pixel_values": torch.stack(tensors)}


def test_paired_collator_returns_float_tensors_labels_and_manifest_metadata() -> None:
    PairedBatchCollator, _, FixedTransformSampler = _task6_types()
    processor = FakeProcessor()
    collator = PairedBatchCollator(
        processor=processor,
        sampler=FixedTransformSampler("noise_s0.10"),
        seed=17,
    )

    batch = collator(_source_items())

    assert batch.clean_pixels.shape == (2, 3, 224, 224)
    assert batch.transformed_pixels.shape == (2, 3, 224, 224)
    assert batch.clean_pixels.dtype == batch.transformed_pixels.dtype == torch.float32
    assert torch.equal(batch.labels, torch.tensor([0.0, 1.0]))
    assert batch.labels.shape == (2,)
    assert batch.sample_ids == ("sample-0", "sample-1")
    assert batch.condition_ids == ("noise_s0.10", "noise_s0.10")
    assert batch.dataset_names == ("sid_set", "wildfake")
    assert batch.generator_families == ("authentic", "sdxl")
    assert batch.source_group_ids == ("source-0", "source-1")
    assert batch.splits == ("train", "train")
    assert batch.split_group_ids == ("split-0", "split-1")
    assert not torch.equal(batch.clean_pixels, batch.transformed_pixels)
    with pytest.raises(FrozenInstanceError):
        batch.labels = torch.zeros(2)  # type: ignore[misc]


def test_collator_epoch_seed_is_repeatable_and_changes_transformed_view() -> None:
    PairedBatchCollator, _, FixedTransformSampler = _task6_types()
    collator = PairedBatchCollator(
        processor=FakeProcessor(),
        sampler=FixedTransformSampler("noise_s0.10"),
        seed=41,
    )
    items = _source_items()

    first = collator(items)
    repeated = collator(items)
    collator.set_epoch(1)
    next_epoch = collator(items)

    assert torch.equal(first.clean_pixels, repeated.clean_pixels)
    assert torch.equal(first.transformed_pixels, repeated.transformed_pixels)
    assert first.condition_ids == repeated.condition_ids
    assert torch.equal(first.clean_pixels, next_epoch.clean_pixels)
    assert not torch.equal(first.transformed_pixels, next_epoch.transformed_pixels)


def test_collator_is_independent_of_batch_order() -> None:
    PairedBatchCollator, _, FixedTransformSampler = _task6_types()
    collator = PairedBatchCollator(
        processor=FakeProcessor(),
        sampler=FixedTransformSampler("noise_s0.05"),
        seed=29,
    )
    collator.set_epoch(4)
    items = _source_items(4)

    forward = collator(items)
    reverse = collator(tuple(reversed(items)))
    forward_by_id = {
        sample_id: (forward.condition_ids[index], forward.transformed_pixels[index])
        for index, sample_id in enumerate(forward.sample_ids)
    }
    reverse_by_id = {
        sample_id: (reverse.condition_ids[index], reverse.transformed_pixels[index])
        for index, sample_id in enumerate(reverse.sample_ids)
    }

    assert set(forward_by_id) == set(reverse_by_id)
    for sample_id in forward_by_id:
        assert forward_by_id[sample_id][0] == reverse_by_id[sample_id][0]
        assert torch.equal(forward_by_id[sample_id][1], reverse_by_id[sample_id][1])


def test_collator_uses_same_processor_twice_after_transform_and_protects_sources() -> None:
    PairedBatchCollator, _, FixedTransformSampler = _task6_types()

    class MutatingFakeProcessor(FakeProcessor):
        def __call__(self, *, images, return_tensors: str):
            result = super().__call__(images=images, return_tensors=return_tensors)
            images[0].putpixel((0, 0), (0, 0, 0))
            return result

    processor = MutatingFakeProcessor()
    items = _source_items()
    originals = tuple(np.asarray(item.image).copy() for item in items)
    collator = PairedBatchCollator(
        processor=processor,
        sampler=FixedTransformSampler("jpeg_q30"),
        seed=17,
    )

    collator(items)

    assert len(processor.calls) == 2
    assert all(len(call) == len(items) for call in processor.calls)
    assert all(
        np.array_equal(np.asarray(item.image), original)
        for item, original in zip(items, originals, strict=True)
    )
    assert any(
        not np.array_equal(clean, transformed)
        for clean, transformed in zip(processor.calls[0], processor.calls[1], strict=True)
    )


@pytest.mark.parametrize("epoch", [-1, True, 1.5, "1"])
def test_collator_rejects_invalid_epochs(epoch: Any) -> None:
    PairedBatchCollator, _, FixedTransformSampler = _task6_types()
    collator = PairedBatchCollator(
        processor=FakeProcessor(),
        sampler=FixedTransformSampler("jpeg_q50"),
        seed=17,
    )

    with pytest.raises(DataIntegrityError, match="epoch.*nonnegative integer"):
        collator.set_epoch(epoch)


def test_collator_rejects_empty_batch() -> None:
    PairedBatchCollator, _, FixedTransformSampler = _task6_types()
    collator = PairedBatchCollator(
        processor=FakeProcessor(),
        sampler=FixedTransformSampler("jpeg_q50"),
        seed=17,
    )

    with pytest.raises(DataIntegrityError, match="nonempty"):
        collator([])


def test_collator_accepts_duplicate_ids_from_replacement_sampler() -> None:
    from prooflens.data.sampling import make_weighted_sampler

    PairedBatchCollator, _, FixedTransformSampler = _task6_types()
    frame = pd.DataFrame(
        [
            {
                "label": 0,
                "dataset_name": "sid_set",
                "generator_family": "authentic",
            },
            {
                "label": 1,
                "dataset_name": "wildfake",
                "generator_family": "sdxl",
            },
        ]
    )
    source_items = _source_items()
    sampled_items = [
        source_items[index]
        for index in make_weighted_sampler(frame, seed=17, num_samples=4)
    ]
    collator = PairedBatchCollator(
        processor=FakeProcessor(),
        sampler=FixedTransformSampler("jpeg_q50"),
        seed=17,
    )

    batch = collator(sampled_items)

    assert batch.clean_pixels.shape == (4, 3, 224, 224)
    assert batch.transformed_pixels.shape == (4, 3, 224, 224)
    assert len(set(batch.sample_ids)) < len(batch.sample_ids)


@pytest.mark.parametrize(
    ("output", "message"),
    [
        ({}, "pixel_values"),
        ({"pixel_values": [[[1.0]]]}, "tensor"),
        ({"pixel_values": torch.zeros(2, 3, 224)}, "rank"),
        ({"pixel_values": torch.zeros(1, 3, 224, 224)}, "batch"),
        ({"pixel_values": torch.zeros(2, 1, 224, 224)}, "shape"),
        ({"pixel_values": torch.zeros(2, 3, 128, 128)}, "shape"),
        ({"pixel_values": torch.zeros(2, 3, 224, 224, dtype=torch.int64)}, "floating"),
        (
            {
                "pixel_values": torch.full(
                    (2, 3, 224, 224), float("nan"), dtype=torch.float32
                )
            },
            "finite",
        ),
    ],
)
def test_collator_rejects_malformed_processor_outputs_with_typed_error(
    output: object, message: str
) -> None:
    PairedBatchCollator, _, FixedTransformSampler = _task6_types()

    class MalformedProcessor:
        def __call__(self, *, images, return_tensors: str):
            return output

    collator = PairedBatchCollator(
        processor=MalformedProcessor(),
        sampler=FixedTransformSampler("jpeg_q50"),
        seed=17,
    )

    with pytest.raises(DataIntegrityError, match=message):
        collator(_source_items())
