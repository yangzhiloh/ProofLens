from __future__ import annotations

import json
from typing import NamedTuple

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from prooflens.errors import DataIntegrityError, ExportError


class TinyOutput(NamedTuple):
    logits: torch.Tensor
    features: torch.Tensor


class TinyDetector(nn.Module):
    def forward(self, pixel_values: torch.Tensor) -> TinyOutput:
        logits = pixel_values.mean(dim=(1, 2, 3)) * 1.25 - 0.2
        features = pixel_values.mean(dim=(2, 3))
        return TinyOutput(logits=logits, features=features)


class FakeProcessor:
    def __call__(self, *, images, return_tensors: str):
        assert return_tensors == "pt"
        values = [sum(image.getpixel((0, 0))) / (3 * 255) for image in images]
        return {
            "pixel_values": torch.stack(
                [torch.full((3, 224, 224), value, dtype=torch.float32) for value in values]
            )
        }


@pytest.fixture(scope="module")
def exported_case(tmp_path_factory):
    from prooflens.export.onnx_export import export_onnx

    output_dir = tmp_path_factory.mktemp("onnx-parity")
    model = TinyDetector().eval()
    sample = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(17))
    onnx_path = export_onnx(model, sample, output_dir / "model.onnx")
    return model, onnx_path, output_dir


def test_onnx_logits_match_pytorch(exported_case) -> None:
    from prooflens.inference.onnx_backend import OnnxTensorBackend

    model, onnx_path, _ = exported_case
    sample = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(29))
    expected = model(sample).logits.detach().numpy()
    actual = OnnxTensorBackend(onnx_path).predict_batch(sample.numpy())

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def test_onnx_dynamic_batch_and_32_sample_probability_parity(exported_case) -> None:
    from prooflens.export.onnx_export import verify_onnx_parity
    from prooflens.inference.onnx_backend import OnnxTensorBackend

    model, onnx_path, output_dir = exported_case
    backend = OnnxTensorBackend(onnx_path)
    one = np.zeros((1, 3, 224, 224), dtype=np.float32)
    parity = torch.randn(32, 3, 224, 224, generator=torch.Generator().manual_seed(41))

    assert backend.predict_batch(one).shape == (1,)
    report_path = output_dir / "export_report.json"
    report = verify_onnx_parity(
        model,
        onnx_path,
        parity,
        temperature=1.7,
        tolerance=1e-4,
        report_path=report_path,
    )
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert report.passed
    assert report.sample_count == 32
    assert report.max_abs_logit_difference <= 1e-4
    assert report.max_abs_probability_difference <= 1e-4
    assert payload["passed"] is True
    assert payload["provider"] == "CPUExecutionProvider"


def test_onnx_image_backend_satisfies_task_12_protocol(exported_case) -> None:
    from prooflens.inference.onnx_backend import OnnxLogitBackend
    from prooflens.inference.service import InferenceService

    _, onnx_path, _ = exported_case
    backend = OnnxLogitBackend(
        onnx_path,
        FakeProcessor(),
        model_version="tiny-onnx-v1",
    )
    service = InferenceService(backend, temperature=1.0)

    prediction = service.predict(Image.new("RGB", (8, 8), (255, 255, 255)))

    assert prediction.logit == pytest.approx(1.05, abs=1e-5)
    assert prediction.model_version == "tiny-onnx-v1"
    assert prediction.preprocessing_version == "dinov2-base-224-v1"


def test_parity_failure_writes_report_and_rejects_publication(
    exported_case, monkeypatch
) -> None:
    import prooflens.export.onnx_export as export_module

    model, onnx_path, output_dir = exported_case
    parity = torch.zeros(32, 3, 224, 224)
    report_path = output_dir / "failed-export-report.json"

    class WrongBackend:
        provider = "CPUExecutionProvider"

        def __init__(self, model_path):
            assert model_path == onnx_path

        def predict_batch(self, values):
            return np.full(values.shape[0], 9.0, dtype=np.float32)

    monkeypatch.setattr(export_module, "OnnxTensorBackend", WrongBackend)

    with pytest.raises(ExportError, match="parity"):
        export_module.verify_onnx_parity(
            model,
            onnx_path,
            parity,
            report_path=report_path,
        )
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is False


@pytest.mark.parametrize(
    "bad_values",
    [
        np.zeros((3, 224, 224), dtype=np.float32),
        np.zeros((0, 3, 224, 224), dtype=np.float32),
        np.full((1, 3, 224, 224), np.nan, dtype=np.float32),
    ],
)
def test_onnx_tensor_backend_rejects_invalid_batches(exported_case, bad_values) -> None:
    from prooflens.inference.onnx_backend import OnnxTensorBackend

    _, onnx_path, _ = exported_case

    with pytest.raises(DataIntegrityError, match="pixel batch"):
        OnnxTensorBackend(onnx_path).predict_batch(bad_values)
