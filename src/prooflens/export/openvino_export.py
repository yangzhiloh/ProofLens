"""Optional OpenVINO conversion; ONNX CPU remains the release fallback."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json


def compile_openvino(onnx_path: Path | str, device: str = "AUTO"):
    import openvino as ov

    core = ov.Core()
    return core.compile_model(str(onnx_path), device)


@dataclass(frozen=True, slots=True)
class OpenVinoReport:
    status: str
    device: str
    error: str | None = None


def smoke_openvino(onnx_path: Path | str, output_path: Path | None = None, device: str = "AUTO") -> OpenVinoReport:
    try:
        compile_openvino(onnx_path, device)
        report = OpenVinoReport("ok", device)
    except Exception as error:
        report = OpenVinoReport("unavailable", device, str(error))
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    return report
