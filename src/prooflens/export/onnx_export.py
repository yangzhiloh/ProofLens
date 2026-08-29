"""ONNX CPU export and numerical parity checks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

import numpy as np
import torch
from torch import Tensor, nn

from prooflens.errors import ExportError


class LogitOnlyWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: Tensor) -> Tensor:
        output = self.model(pixel_values)
        logits = output.logits if hasattr(output, "logits") else output
        return logits


@dataclass(frozen=True, slots=True)
class ParityReport:
    samples: int
    max_absolute_logit_difference: float
    max_absolute_probability_difference: float
    tolerance: float
    passed: bool


def export_onnx(model: nn.Module, sample_pixels: Tensor, onnx_path: Path) -> Path:
    if not isinstance(sample_pixels, Tensor) or sample_pixels.ndim != 4:
        raise ExportError("ONNX export sample must be a rank-4 tensor")
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = LogitOnlyWrapper(model).eval()
    sample = sample_pixels.detach().cpu().to(dtype=torch.float32)
    try:
        with redirect_stdout(StringIO()):
            torch.onnx.export(
                wrapper,
                (sample,),
                onnx_path,
                input_names=["pixel_values"],
                output_names=["logits"],
                dynamo=True,
                dynamic_shapes={"pixel_values": {0: torch.export.Dim("batch", min=1, max=32)}},
                opset_version=18,
            )
    except Exception as first_error:
        try:
            with redirect_stdout(StringIO()):
                torch.onnx.export(
                    wrapper,
                    (sample,),
                    onnx_path,
                    input_names=["pixel_values"],
                    output_names=["logits"],
                    dynamic_axes={"pixel_values": {0: "batch"}, "logits": {0: "batch"}},
                    opset_version=18,
                    dynamo=False,
                )
        except Exception as error:
            raise ExportError(f"ONNX export failed: {error}") from first_error
    return onnx_path


def verify_onnx_parity(
    model: nn.Module,
    onnx_path: Path,
    parity_pixels: Tensor,
    tolerance: float = 1e-4,
) -> ParityReport:
    from prooflens.inference.onnx_backend import OnnxTensorBackend

    values = parity_pixels.detach().cpu().numpy().astype(np.float32)
    model.eval()
    with torch.no_grad():
        expected = np.asarray(model(parity_pixels).logits.detach().cpu())
    actual = OnnxTensorBackend(onnx_path).predict_batch(values)
    logit_difference = np.abs(actual - expected)
    probability_difference = np.abs(_sigmoid(actual) - _sigmoid(expected))
    report = ParityReport(
        samples=int(values.shape[0]),
        max_absolute_logit_difference=float(logit_difference.max(initial=0.0)),
        max_absolute_probability_difference=float(probability_difference.max(initial=0.0)),
        tolerance=float(tolerance),
        passed=bool(probability_difference.max(initial=0.0) <= tolerance),
    )
    if not report.passed:
        raise ExportError(
            "ONNX parity exceeded tolerance: "
            f"max probability difference={report.max_absolute_probability_difference:.8f}"
        )
    return report


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))


# Compatibility exports keep tensor-runtime access discoverable from the export
# module while the implementation remains in the inference namespace.
from prooflens.inference.onnx_backend import OnnxLogitBackend, OnnxTensorBackend  # noqa: E402,F401
