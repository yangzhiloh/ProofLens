import torch

from prooflens.training.hard_mining import HardTransformMiner, select_lowest_margin
from prooflens.data.transforms import canonical_specs


def test_miner_selects_lowest_correct_margin() -> None:
    logits_by_condition = {
        "jpeg_q30": torch.tensor([2.0, -2.0]),
        "blur_s2.0": torch.tensor([-1.0, 1.0]),
        "noise_s0.10": torch.tensor([0.5, -0.5]),
    }
    assert select_lowest_margin(logits_by_condition, torch.tensor([1.0, 0.0])) == ("blur_s2.0", "blur_s2.0")


def test_candidate_sampling_uses_distinct_families() -> None:
    miner = HardTransformMiner(canonical_specs(), seed=17, candidate_count=3)
    candidates = miner.sample_candidates(("a", "b"), epoch=1)
    assert len(candidates) == 2
    assert all(len({spec.family for spec in row}) == 3 for row in candidates)
