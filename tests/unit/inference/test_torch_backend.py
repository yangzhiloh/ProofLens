from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from prooflens.errors import DataIntegrityError, TrainingError


class FakeProcessor:
    def __call__(self, *, images, return_tensors: str):
        assert return_tensors == "pt"
        values = [sum(image.getpixel((0, 0))) / (3 * 255) for image in images]
        return {
            "pixel_values": torch.stack(
                [torch.full((3, 224, 224), value, dtype=torch.float32) for value in values]
            )
        }


class TinyDetector(nn.Module):
    def __init__(self, bias: float = 0.0) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias))

    def forward(self, pixel_values: torch.Tensor):
        logits = pixel_values.mean(dim=(1, 2, 3)) + self.bias
        return SimpleNamespace(logits=logits)


def test_torch_backend_runs_shared_preprocessing_and_returns_one_cpu_logit() -> None:
    from prooflens.inference.torch_backend import TorchLogitBackend

    model = TinyDetector(bias=0.25)
    backend = TorchLogitBackend(
        model,
        FakeProcessor(),
        model_version="tiny-v1",
    )

    logit = backend.predict_logit(Image.new("RGB", (4, 4), (255, 255, 255)))

    assert logit == pytest.approx(1.25)
    assert backend.device.type == "cpu"
    assert backend.model_version == "tiny-v1"
    assert not model.training


@pytest.mark.parametrize(
    "output_factory",
    [
        lambda: torch.tensor([0.75]),
        lambda: {"logits": torch.tensor([0.75])},
        lambda: {"logit": torch.tensor(0.75)},
        lambda: SimpleNamespace(logit=torch.tensor([0.75])),
    ],
)
def test_torch_backend_accepts_supported_single_logit_outputs(output_factory) -> None:
    from prooflens.inference.torch_backend import TorchLogitBackend

    class OutputModel(nn.Module):
        def forward(self, pixel_values: torch.Tensor):
            return output_factory()

    backend = TorchLogitBackend(OutputModel(), FakeProcessor(), model_version="output-v1")

    assert backend.predict_logit(Image.new("RGB", (2, 2))) == pytest.approx(0.75)


@pytest.mark.parametrize(
    "bad_output",
    [torch.tensor([1.0, 2.0]), {"unknown": torch.tensor([1.0])}, "bad"],
)
def test_torch_backend_rejects_malformed_model_outputs(bad_output) -> None:
    from prooflens.inference.torch_backend import TorchLogitBackend

    class BadModel(nn.Module):
        def forward(self, pixel_values: torch.Tensor):
            return bad_output

    backend = TorchLogitBackend(BadModel(), FakeProcessor(), model_version="bad-v1")

    with pytest.raises(DataIntegrityError, match="single finite logit"):
        backend.predict_logit(Image.new("RGB", (2, 2)))


def test_cuda_is_used_only_when_explicitly_requested_and_available(monkeypatch) -> None:
    from prooflens.inference.torch_backend import resolve_torch_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_torch_device(None).type == "cpu"
    assert resolve_torch_device("auto").type == "cpu"
    assert resolve_torch_device("cpu").type == "cpu"
    assert resolve_torch_device("cuda").type == "cpu"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_torch_device("cuda").type == "cuda"


def test_torch_backend_loads_an_injected_model_from_checkpoint(tmp_path) -> None:
    from prooflens.inference.torch_backend import TorchLogitBackend

    checkpoint = tmp_path / "selected.pt"
    trained = TinyDetector(bias=1.5)
    torch.save(
        {"model_state": trained.state_dict(), "model_version": "selected-run-v2"},
        checkpoint,
    )

    backend = TorchLogitBackend.from_checkpoint(
        checkpoint,
        model_factory=TinyDetector,
        processor=FakeProcessor(),
    )

    assert backend.model_version == "selected-run-v2"
    assert backend.predict_logit(Image.new("RGB", (2, 2))) == pytest.approx(1.5)


def test_torch_backend_reports_missing_or_invalid_checkpoints(tmp_path) -> None:
    from prooflens.inference.torch_backend import TorchLogitBackend

    with pytest.raises(TrainingError, match="does not exist"):
        TorchLogitBackend.from_checkpoint(
            tmp_path / "missing.pt",
            model_factory=TinyDetector,
            processor=FakeProcessor(),
        )

    invalid = tmp_path / "invalid.pt"
    torch.save({"optimizer_state": {}}, invalid)
    with pytest.raises(TrainingError, match="model state"):
        TorchLogitBackend.from_checkpoint(
            invalid,
            model_factory=TinyDetector,
            processor=FakeProcessor(),
        )


def test_torch_backend_wraps_restricted_checkpoint_load_errors(tmp_path) -> None:
    from prooflens.inference.torch_backend import TorchLogitBackend

    checkpoint = tmp_path / "legacy-unsafe.pt"
    torch.save(
        {
            "model": TinyDetector().state_dict(),
            "numpy_rng": np.random.get_state(),
        },
        checkpoint,
    )

    with pytest.raises(TrainingError, match="could not be loaded"):
        TorchLogitBackend.from_checkpoint(
            checkpoint,
            model_factory=TinyDetector,
            processor=FakeProcessor(),
        )
