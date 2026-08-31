from __future__ import annotations

import gc
import json
import math
import subprocess
import sys
import warnings
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from torch import Tensor, nn

from prooflens.errors import ExportError
from prooflens.inference.onnx_backend import OnnxTensorBackend


@dataclass(frozen=True, slots=True)
class ExportParityReport:
    onnx_path: Path
    provider: str
    sample_count: int
    temperature: float
    tolerance: float
    max_abs_logit_difference: float
    max_abs_probability_difference: float
    passed: bool


class LogitOnlyWrapper(nn.Module):
    """Expose only the detector's binary logits to the exported graph."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: Tensor) -> Tensor:
        return _extract_logits(self.model(pixel_values))


def export_onnx(model: nn.Module, sample_pixels: Tensor, onnx_path: Path) -> Path:
    """Export a dynamic-batch opset-18 ONNX model atomically.

    The legacy exporter is intentional here: the pinned PyTorch 2.5 runtime has
    a Dynamo exporter failure when a model reshapes with a symbolic batch size.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("ONNX export model must be a torch.nn.Module")
    pixels = _validated_pixel_tensor(sample_pixels, minimum_batch=1)
    if pixels.shape[0] > 32:
        raise ExportError("ONNX export sample batch cannot exceed 32 images")
    destination = Path(onnx_path)
    if destination.suffix.lower() != ".onnx":
        raise ExportError("ONNX export destination must use the .onnx suffix")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(f".{destination.stem}.{uuid4().hex}.onnx")
    wrapper = LogitOnlyWrapper(model).eval()
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`isinstance\(treespec, LeafSpec\)` is deprecated.*",
                category=FutureWarning,
            )
            torch.onnx.export(
                wrapper,
                (pixels,),
                temporary_path,
                input_names=["pixel_values"],
                output_names=["logits"],
                dynamo=False,
                dynamic_axes={
                    "pixel_values": {0: "batch"},
                    "logits": {0: "batch"},
                },
                opset_version=18,
                external_data=False,
            )
        _check_onnx_file(temporary_path)
        temporary_path.replace(destination)
    except Exception as error:
        raise ExportError(f"ONNX export failed for {destination}") from error
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return destination


def _check_onnx_file(path: Path) -> None:
    """Run the native ONNX checker outside the PyTorch process on Windows.

    Loading PyTorch and ONNX's native checker into one Windows process can cause
    a DLL-level access violation for production-sized graphs. A clean child
    process preserves full ONNX validation without weakening the publication gate.
    """

    command = [
        sys.executable,
        "-c",
        "import onnx,sys; onnx.checker.check_model(sys.argv[1])",
        str(path),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "native checker failed"
        raise ExportError(f"ONNX checker rejected {path}: {detail}")


def verify_onnx_parity(
    model: nn.Module,
    onnx_path: Path,
    parity_pixels: Tensor,
    *,
    temperature: float = 1.0,
    tolerance: float = 1e-4,
    report_path: Path | None = None,
    release_model_before_onnx: bool = False,
) -> ExportParityReport:
    """Require 32-image logit and calibrated-probability parity before publication."""

    if not isinstance(model, nn.Module):
        raise TypeError("parity model must be a torch.nn.Module")
    pixels = _validated_pixel_tensor(parity_pixels, minimum_batch=32)
    if pixels.shape[0] > 32:
        raise ExportError("parity batch cannot exceed the exported maximum of 32 images")
    numeric_temperature = _positive_finite(temperature, "temperature")
    numeric_tolerance = _positive_finite(tolerance, "tolerance")
    wrapper = LogitOnlyWrapper(model).eval()
    chunk_size = 4
    with torch.inference_mode():
        expected = np.concatenate(
            [
                wrapper(chunk)
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .numpy()
                .reshape(-1)
                for chunk in pixels.split(chunk_size)
            ]
        )
    if release_model_before_onnx:
        model.to(device="meta")
        del wrapper
        gc.collect()
    backend = OnnxTensorBackend(Path(onnx_path))
    pixel_values = pixels.detach().to(device="cpu", dtype=torch.float32).numpy()
    actual = np.concatenate(
        [
            backend.predict_batch(pixel_values[start : start + chunk_size]).reshape(-1)
            for start in range(0, len(pixel_values), chunk_size)
        ]
    )
    if actual.shape != expected.shape:
        raise ExportError(
            f"ONNX parity output shape {actual.shape} does not match PyTorch {expected.shape}"
        )
    max_logit_difference = float(np.max(np.abs(actual - expected)))
    expected_probability = _sigmoid_array(expected / numeric_temperature)
    actual_probability = _sigmoid_array(actual / numeric_temperature)
    max_probability_difference = float(
        np.max(np.abs(actual_probability - expected_probability))
    )
    passed = (
        max_logit_difference <= numeric_tolerance
        and max_probability_difference <= numeric_tolerance
    )
    report = ExportParityReport(
        onnx_path=Path(onnx_path),
        provider=backend.provider,
        sample_count=int(pixels.shape[0]),
        temperature=numeric_temperature,
        tolerance=numeric_tolerance,
        max_abs_logit_difference=max_logit_difference,
        max_abs_probability_difference=max_probability_difference,
        passed=passed,
    )
    destination = report_path or Path(onnx_path).with_name("export_report.json")
    _write_parity_report(report, Path(destination))
    if not passed:
        raise ExportError(
            "ONNX parity failed: maximum absolute logit difference "
            f"{max_logit_difference:.8f}, calibrated probability difference "
            f"{max_probability_difference:.8f}, tolerance {numeric_tolerance:.8f}"
        )
    return report


def _extract_logits(output: Any) -> Tensor:
    candidate: object
    if isinstance(output, Tensor):
        candidate = output
    elif isinstance(output, Mapping):
        candidate = output.get("logits", output.get("logit"))
    elif hasattr(output, "logits"):
        candidate = output.logits
    elif hasattr(output, "logit"):
        candidate = output.logit
    else:
        candidate = None
    if not isinstance(candidate, Tensor):
        raise ExportError("detector output must expose a logits tensor")
    if candidate.ndim == 0:
        candidate = candidate.unsqueeze(0)
    if candidate.ndim != 1:
        raise ExportError("detector logits must have shape [batch]")
    return candidate


def _validated_pixel_tensor(value: object, minimum_batch: int) -> Tensor:
    if not isinstance(value, Tensor):
        raise ExportError("pixel batch must be a Torch tensor")
    if value.ndim != 4 or value.shape[0] < minimum_batch:
        raise ExportError(
            f"pixel batch must have rank 4 and contain at least {minimum_batch} images"
        )
    if not value.is_floating_point() or not torch.isfinite(value).all().item():
        raise ExportError("pixel batch must contain finite floating values")
    return value.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _positive_finite(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{field} must be a finite positive number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"{field} must be a finite positive number")
    return numeric


def _write_parity_report(report: ExportParityReport, path: Path) -> None:
    payload = asdict(report)
    payload["onnx_path"] = str(report.onnx_path)
    _atomic_write_text(
        path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(text, encoding="utf-8")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
