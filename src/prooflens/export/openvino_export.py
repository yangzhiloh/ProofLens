from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np


@dataclass(frozen=True, slots=True)
class OpenVinoSmokeReport:
    success: bool
    device: str
    first_prediction: float | None
    error: str | None
    onnx_path: Path


def compile_openvino(onnx_path: Path, device: str = "AUTO") -> Any:
    """Compile ONNX lazily so OpenVINO remains an optional dependency."""

    path = Path(onnx_path)
    if not path.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {path}")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("OpenVINO device must be a nonempty string")
    import openvino as ov

    core = ov.Core()
    return core.compile_model(path, device.strip())


def try_openvino_smoke(
    onnx_path: Path,
    sample_pixels: np.ndarray,
    report_path: Path,
    *,
    device: str = "AUTO",
) -> OpenVinoSmokeReport:
    """Attempt optional compilation and always preserve a machine-readable result."""

    path = Path(onnx_path)
    try:
        values = np.asarray(sample_pixels, dtype=np.float32)
        if values.ndim != 4 or values.shape[0] < 1 or not np.isfinite(values).all():
            raise ValueError("OpenVINO smoke pixels must be a finite nonempty rank-four batch")
        compiled = compile_openvino(path, device)
        outputs = compiled([values])
        first_output = np.asarray(next(iter(outputs.values())), dtype=np.float32).reshape(-1)
        if first_output.size < 1 or not np.isfinite(first_output).all():
            raise ValueError("OpenVINO smoke output must contain a finite prediction")
        report = OpenVinoSmokeReport(
            success=True,
            device=device,
            first_prediction=float(first_output[0]),
            error=None,
            onnx_path=path,
        )
    except Exception as error:  # noqa: BLE001 - optional acceleration must remain non-blocking.
        report = OpenVinoSmokeReport(
            success=False,
            device=device,
            first_prediction=None,
            error=f"{type(error).__name__}: {error}",
            onnx_path=path,
        )
    _write_report(report, Path(report_path))
    return report


def _write_report(report: OpenVinoSmokeReport, path: Path) -> None:
    payload = asdict(report)
    payload["onnx_path"] = str(report.onnx_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
