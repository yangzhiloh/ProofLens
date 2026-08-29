from __future__ import annotations

import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

from prooflens.data.transforms import group_specs_by_family
from prooflens.errors import DataIntegrityError


def _sampling_module():
    from prooflens.data import sampling

    return sampling


def _imbalanced_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": [0, 0, 0, 1, 1, 1, 1, 1, 1],
            "dataset_name": ["sid", "sid", "camera", "wild", "wild", "wild", "wild", "other", "other"],
            "generator_family": [
                "authentic",
                "authentic",
                "authentic",
                "sdxl",
                "sdxl",
                "sdxl",
                "flux",
                "sdxl",
                "sdxl",
            ],
        },
        index=[11, 11, -4, 100, 2, 2, 7, 7, 7],
    )


def test_sampling_weights_assign_exact_equal_label_and_stratum_mass_positionally() -> None:
    frame = _imbalanced_frame()
    original = frame.copy(deep=True)

    weights = _sampling_module().compute_sampling_weights(frame)

    assert isinstance(weights, np.ndarray)
    assert weights.dtype == np.float64
    assert weights.tolist() == pytest.approx(
        [1 / 8, 1 / 8, 1 / 4, 1 / 18, 1 / 18, 1 / 18, 1 / 6, 1 / 12, 1 / 12]
    )
    positional = frame.reset_index(drop=True).assign(weight=weights)
    assert positional.groupby("label")["weight"].sum().to_dict() == pytest.approx(
        {0: 0.5, 1: 0.5}
    )
    fake_mass = (
        positional[positional.label == 1]
        .groupby(["dataset_name", "generator_family"])["weight"]
        .sum()
    )
    assert fake_mass.to_numpy() == pytest.approx([1 / 6, 1 / 6, 1 / 6])
    assert np.isfinite(weights).all()
    assert (weights > 0).all()
    assert weights.sum() == pytest.approx(1.0)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    ("row", "column"),
    [(0, "dataset_name"), (3, "dataset_name"), (3, "generator_family")],
)
def test_sampling_weights_reject_blank_stratum_metadata(row: int, column: str) -> None:
    frame = _imbalanced_frame()
    frame.iloc[row, frame.columns.get_loc(column)] = "  "

    with pytest.raises(DataIntegrityError, match=column):
        _sampling_module().compute_sampling_weights(frame)


@pytest.mark.parametrize("bad_label", [2, -1, True, "1", 0.5, None])
def test_sampling_weights_reject_nonbinary_labels(bad_label: Any) -> None:
    frame = _imbalanced_frame()
    frame["label"] = frame["label"].astype(object)
    frame.iloc[0, frame.columns.get_loc("label")] = bad_label

    with pytest.raises(DataIntegrityError, match="label.*binary"):
        _sampling_module().compute_sampling_weights(frame)


def test_sampling_weights_reject_single_label_frames() -> None:
    frame = _imbalanced_frame().loc[lambda value: value.label == 1]

    with pytest.raises(DataIntegrityError, match="both labels"):
        _sampling_module().compute_sampling_weights(frame)


def test_weighted_sampler_is_seeded_repeatable_and_isolated_from_global_rng() -> None:
    sampling = _sampling_module()
    frame = _imbalanced_frame()
    torch.manual_seed(991)
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()

    first = list(sampling.make_weighted_sampler(frame, num_samples=40, seed=17))
    second = list(sampling.make_weighted_sampler(frame, num_samples=40, seed=17))
    third = list(sampling.make_weighted_sampler(frame, num_samples=40, seed=29))

    assert first == second
    assert first != third
    assert len(first) == 40
    assert all(0 <= index < len(frame) for index in first)
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    after_numpy = np.random.get_state()
    assert after_numpy[0] == numpy_state[0]
    assert np.array_equal(after_numpy[1], numpy_state[1])
    assert after_numpy[2:] == numpy_state[2:]


def test_stable_seed_has_process_independent_typed_test_vectors() -> None:
    stable_seed = _sampling_module().stable_seed

    assert stable_seed(17, 2, "sample-a") == 7904897493114868749
    assert stable_seed(None, False, True, -7, 3.5, "λ", b"\x00") == 4422597127991316094
    assert 0 <= stable_seed("range") < 2**63

    project_root = Path(__file__).parents[3]
    code = "from prooflens.data.sampling import stable_seed; print(stable_seed(17, 2, 'sample-a'))"
    observed = []
    for hash_seed in ("1", "987654"):
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONHASHSEED": hash_seed,
                "PYTHONPATH": str(project_root / "src"),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        observed.append(
            subprocess.check_output(
                [sys.executable, "-c", code],
                cwd=project_root,
                env=environment,
                text=True,
            ).strip()
        )
    assert observed == ["7904897493114868749", "7904897493114868749"]


def test_stable_seed_separates_types_and_ambiguous_component_boundaries() -> None:
    stable_seed = _sampling_module().stable_seed

    values = {
        stable_seed("1"),
        stable_seed(1),
        stable_seed(True),
        stable_seed("ab", "c"),
        stable_seed("a", "bc"),
        stable_seed(b"1"),
    }
    assert len(values) == 6


@pytest.mark.parametrize("component", [object(), [], {}, float("nan"), float("inf")])
def test_stable_seed_rejects_unsupported_or_nonfinite_components(component: object) -> None:
    with pytest.raises(DataIntegrityError, match="stable seed component"):
        _sampling_module().stable_seed(component)


def test_fixed_and_family_balanced_transform_samplers_follow_the_typed_contract() -> None:
    sampling = _sampling_module()
    fixed = sampling.FixedTransformSampler("jpeg_q50")
    balanced = sampling.FamilyBalancedTransformSampler()

    assert fixed.sample("sample", epoch=9, seed=17).condition_id == "jpeg_q50"
    first = {
        sample_id: balanced.sample(sample_id, epoch=3, seed=17).condition_id
        for sample_id in (f"sample-{index}" for index in range(128))
    }
    reordered = {
        sample_id: balanced.sample(sample_id, epoch=3, seed=17).condition_id
        for sample_id in reversed(tuple(first))
    }
    next_epoch = {
        sample_id: balanced.sample(sample_id, epoch=4, seed=17).condition_id
        for sample_id in first
    }
    next_seed = {
        sample_id: balanced.sample(sample_id, epoch=3, seed=29).condition_id
        for sample_id in first
    }

    assert reordered == first
    assert any(next_epoch[sample_id] != first[sample_id] for sample_id in first)
    assert any(next_seed[sample_id] != first[sample_id] for sample_id in first)


def test_family_balanced_sampler_is_uniform_by_family_then_severity() -> None:
    sampler = _sampling_module().FamilyBalancedTransformSampler()
    grouped = group_specs_by_family()
    family_by_id = {
        spec.condition_id: family for family, specs in grouped.items() for spec in specs
    }
    selections = [
        sampler.sample(f"distribution-{index}", epoch=5, seed=17)
        for index in range(12_000)
    ]
    families = Counter(spec.family for spec in selections)
    conditions = Counter(spec.condition_id for spec in selections)

    for family, specs in grouped.items():
        assert families[family] / len(selections) == pytest.approx(1 / 6, abs=0.015)
        for spec in specs:
            conditional = conditions[spec.condition_id] / families[family_by_id[spec.condition_id]]
            assert conditional == pytest.approx(1 / len(specs), abs=0.035)
