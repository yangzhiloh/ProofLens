from pathlib import Path

import numpy as np
import torch
from torch import nn

from prooflens.export.onnx_export import export_onnx
from prooflens.inference.onnx_backend import OnnxTensorBackend
from prooflens.models.types import DetectorOutput


class TinyDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Conv2d(3, 1, kernel_size=1)

    def forward(self, pixel_values):
        logits = self.layer(pixel_values).mean(dim=(1, 2, 3))
        return DetectorOutput(logits=logits, features=torch.ones((pixel_values.shape[0], 1)))


def test_onnx_logits_match_pytorch(tmp_path: Path) -> None:
    model = TinyDetector().eval()
    sample = torch.randn(2, 3, 28, 28)
    onnx_path = export_onnx(model, sample, tmp_path / "model.onnx")
    expected = model(sample).logits.detach().numpy()
    actual = OnnxTensorBackend(onnx_path).predict_batch(sample.numpy())
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
