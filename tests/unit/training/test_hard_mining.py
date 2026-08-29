from __future__ import annotations

import logging

import pytest
import torch
from PIL import Image
from torch import nn

from prooflens.data.dataset import SourceItem
from prooflens.models.types import DetectorOutput


class FakeProcessor:
    def __call__(self, *, images, return_tensors: str):
        assert return_tensors == "pt"
        values = [torch.full((3, 224, 224), image.getpixel((0, 0))[0] / 255) for image in images]
        return {"pixel_values": torch.stack(values)}


class RecordingDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.grad_modes: list[bool] = []

    def forward(self, pixels: torch.Tensor) -> DetectorOutput:
        self.grad_modes.append(torch.is_grad_enabled())
        logits = pixels.mean(dim=(1, 2, 3)) * self.scale
        features = torch.stack((logits, logits + 1), dim=1)
        return DetectorOutput(logits=logits, features=features)


def test_select_lowest_margin_uses_each_samples_correct_class() -> None:
    from prooflens.training.hard_mining import select_lowest_margin

    logits_by_condition = {
        "jpeg_q30": torch.tensor([2.0, -2.0]),
        "blur_s2.0": torch.tensor([-1.0, 1.0]),
        "noise_s0.10": torch.tensor([0.5, -0.5]),
    }

    selection = select_lowest_margin(
        logits_by_condition, torch.tensor([1.0, 0.0])
    )

    assert selection == ("blur_s2.0", "blur_s2.0")


def test_candidate_sampling_is_repeatable_and_uses_distinct_families() -> None:
    from prooflens.data.transforms import canonical_specs
    from prooflens.training.hard_mining import HardTransformMiner

    miner = HardTransformMiner(canonical_specs(), seed=17, candidate_count=3)

    first = miner.sample_candidates(("a", "b"), epoch=2)
    repeated = miner.sample_candidates(("a", "b"), epoch=2)

    assert first == repeated
    assert all(len({spec.family for spec in row}) == 3 for row in first)


def test_miner_selects_lowest_margin_per_sample() -> None:
    from prooflens.data.transforms import canonical_specs
    from prooflens.training.hard_mining import HardTransformMiner

    miner = HardTransformMiner(
        canonical_specs(), seed=17, candidate_count=3, exploration_probability=0.0
    )
    condition_ids = (
        ("jpeg_q30", "blur_s2.0", "noise_s0.10"),
        ("jpeg_q30", "blur_s2.0", "noise_s0.10"),
    )
    logits = torch.tensor([[2.0, -1.0, 0.5], [-2.0, 1.0, -0.5]])

    selected = miner.select(
        logits,
        condition_ids,
        torch.tensor([1.0, 0.0]),
        ("a", "b"),
        epoch=3,
    )

    assert selected == ("blur_s2.0", "blur_s2.0")


def test_exploration_is_repeatable_and_can_override_hardest() -> None:
    from prooflens.data.transforms import canonical_specs
    from prooflens.training.hard_mining import HardTransformMiner

    miner = HardTransformMiner(
        canonical_specs(), seed=31, candidate_count=3, exploration_probability=1.0
    )
    conditions = (("jpeg_q30", "blur_s2.0", "noise_s0.10"),)
    logits = torch.tensor([[-10.0, 5.0, 5.0]])

    first = miner.select(logits, conditions, torch.tensor([1.0]), ("sample",), 4)
    repeated = miner.select(logits, conditions, torch.tensor([1.0]), ("sample",), 4)

    assert first == repeated
    assert first != ("jpeg_q30",)


def test_family_proportions_warn_when_selection_collapses(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from prooflens.data.transforms import canonical_specs
    from prooflens.training.hard_mining import HardTransformMiner

    miner = HardTransformMiner(
        canonical_specs(), seed=17, candidate_count=3, exploration_probability=0.0
    )
    conditions = tuple(
        ("jpeg_q30", "blur_s2.0", "noise_s0.10") for _ in range(4)
    )
    logits = torch.tensor([[-5.0, 1.0, 2.0]] * 4)
    miner.select(logits, conditions, torch.ones(4), tuple("abcd"), epoch=1)

    with caplog.at_level(logging.WARNING):
        proportions = miner.epoch_family_proportions(reset=True)

    assert proportions["selected"]["jpeg"] == pytest.approx(1.0)
    assert "exceeds 60 percent" in caplog.text
    assert miner.epoch_family_proportions()["selected"] == {}


def test_hard_mining_candidates_are_no_grad_and_selected_view_backpropagates() -> None:
    from prooflens.data.transforms import canonical_specs
    from prooflens.training.hard_mining import (
        HardMiningCollator,
        HardTransformMiner,
        compute_hard_mined_loss,
    )

    items = tuple(
        SourceItem(
            image=Image.new("RGB", (8, 8), (50 + index * 100, 20, 10)),
            label=index,
            sample_id=f"sample-{index}",
            dataset_name="fixture",
            generator_family="authentic" if index == 0 else "generator",
            source_group_id=f"source-{index}",
            split="train",
            split_group_id=f"group-{index}",
        )
        for index in range(2)
    )
    processor = FakeProcessor()
    batch = HardMiningCollator(processor=processor)(items)
    model = RecordingDetector()
    miner = HardTransformMiner(
        canonical_specs(), seed=17, candidate_count=3, exploration_probability=0.0
    )

    result = compute_hard_mined_loss(
        model=model,
        batch=batch,
        processor=processor,
        miner=miner,
        epoch=1,
        device="cpu",
    )
    result.total.backward()

    assert model.grad_modes == [False, True, True]
    assert model.scale.grad is not None
    assert result.total.isfinite()
