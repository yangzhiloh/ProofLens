from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from prooflens.errors import DataIntegrityError, ExportError
from prooflens.inference.preprocess import (
    PREPROCESSING_VERSION,
    ImageProcessor,
    preprocess_images,
)


class OnnxTensorBackend:
    """Run exported pixel tensors using the mandatory CPU execution provider."""

    provider = "CPUExecutionProvider"

    def __init__(self, model_path: Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise ExportError(f"ONNX model does not exist: {path}")
        try:
            self.session = ort.InferenceSession(
                str(path), providers=[self.provider]
            )
        except Exception as error:
            raise ExportError(f"ONNX Runtime could not load model: {path}") from error
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or not outputs:
            raise ExportError("ONNX model must expose one pixel input and at least one output")
        self.input_name = inputs[0].name
        self.output_name = next(
            (output.name for output in outputs if output.name == "logits"), outputs[0].name
        )

    def predict_batch(self, pixel_values: np.ndarray) -> np.ndarray:
        """Return one finite logit per item from a nonempty rank-four pixel batch."""

        try:
            values = np.asarray(pixel_values, dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise DataIntegrityError("ONNX pixel batch must contain numeric values") from error
        if values.ndim != 4 or values.shape[0] < 1:
            raise DataIntegrityError("ONNX pixel batch must be nonempty with rank 4")
        if not np.isfinite(values).all():
            raise DataIntegrityError("ONNX pixel batch must contain only finite values")
        try:
            output = self.session.run([self.output_name], {self.input_name: values})[0]
        except Exception as error:
            raise DataIntegrityError("ONNX Runtime inference failed for the pixel batch") from error
        logits = np.asarray(output, dtype=np.float32).reshape(-1)
        if logits.shape != (values.shape[0],) or not np.isfinite(logits).all():
            raise DataIntegrityError("ONNX output must contain one finite logit per image")
        return logits


class OnnxLogitBackend:
    """Image-level ONNX adapter satisfying the Task 12 logit protocol."""

    def __init__(
        self,
        model_path: Path,
        processor: ImageProcessor,
        *,
        model_version: str,
        preprocessing_version: str = PREPROCESSING_VERSION,
    ) -> None:
        if not callable(processor):
            raise TypeError("processor must be callable")
        self.model_version = _nonempty_text(model_version, "model_version")
        self.preprocessing_version = _nonempty_text(
            preprocessing_version, "preprocessing_version"
        )
        self.tensor_backend = OnnxTensorBackend(model_path)
        self.processor = processor

    def predict_logit(self, image: Image.Image) -> float:
        pixel_values = preprocess_images((image,), processor=self.processor)
        values = pixel_values.detach().to(device="cpu").numpy()
        return float(self.tensor_backend.predict_batch(values)[0])


def _nonempty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value
