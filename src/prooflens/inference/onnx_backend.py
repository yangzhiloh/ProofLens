"""Required CPU ONNX Runtime backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


class OnnxTensorBackend:
    def __init__(self, model_path: Path | str) -> None:
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict_batch(self, pixel_values: np.ndarray) -> np.ndarray:
        values = np.asarray(pixel_values, dtype=np.float32)
        if values.ndim != 4:
            raise ValueError("ONNX pixel_values must be rank 4")
        return np.asarray(self.session.run([self.output_name], {self.input_name: values})[0])


class OnnxLogitBackend:
    def __init__(self, model_path: Path | str, processor, model_version: str) -> None:
        self.tensor_backend = OnnxTensorBackend(model_path)
        self.processor = processor
        self.model_version = model_version

    def predict_logit(self, image: Image.Image) -> float:
        output = self.processor(images=[image], return_tensors="np")
        pixel_values = output["pixel_values"]
        if hasattr(pixel_values, "detach"):
            pixel_values = pixel_values.detach().cpu().numpy()
        return float(self.tensor_backend.predict_batch(pixel_values).reshape(-1)[0])
