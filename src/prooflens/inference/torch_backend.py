from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import Tensor, nn

from prooflens.errors import DataIntegrityError, TrainingError
from prooflens.inference.preprocess import (
    PREPROCESSING_VERSION,
    ImageProcessor,
    create_dinov2_processor,
    preprocess_images,
)


class TorchLogitBackend:
    """CPU-first PyTorch adapter over the shared preprocessing contract."""

    def __init__(
        self,
        model: nn.Module,
        processor: ImageProcessor,
        *,
        model_version: str,
        device: str | torch.device | None = None,
        preprocessing_version: str = PREPROCESSING_VERSION,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if not callable(processor):
            raise TypeError("processor must be callable")
        self.model_version = _nonempty_text(model_version, "model_version")
        self.preprocessing_version = _nonempty_text(
            preprocessing_version, "preprocessing_version"
        )
        self.device = resolve_torch_device(device)
        self.model = model.to(self.device).eval()
        self.processor = processor

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        *,
        model_factory: Callable[[], nn.Module],
        processor: ImageProcessor | None = None,
        model_version: str | None = None,
        device: str | torch.device | None = None,
    ) -> TorchLogitBackend:
        """Restore an injected model factory from a selected atomic checkpoint."""

        path = Path(checkpoint_path)
        if not path.is_file():
            raise TrainingError(f"checkpoint does not exist: {path}")
        resolved_device = resolve_torch_device(device)
        try:
            payload = torch.load(path, map_location=resolved_device, weights_only=True)
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise TrainingError(f"checkpoint could not be loaded: {path}") from error
        state = _extract_model_state(payload)
        try:
            model = model_factory()
        except Exception as error:
            raise TrainingError("model factory could not construct the checkpoint model") from error
        if not isinstance(model, nn.Module):
            raise TrainingError("model factory must return a torch.nn.Module")
        try:
            model.load_state_dict(state, strict=True)
        except (RuntimeError, TypeError, ValueError) as error:
            raise TrainingError("checkpoint model state is incompatible with the model") from error
        checkpoint_version = _checkpoint_model_version(payload)
        selected_version = model_version or checkpoint_version or path.stem
        selected_processor = processor or create_dinov2_processor()
        return cls(
            model,
            selected_processor,
            model_version=selected_version,
            device=resolved_device,
        )

    def predict_logit(self, image: Image.Image) -> float:
        """Preprocess one image and return one finite scalar logit."""

        pixel_values = preprocess_images((image,), processor=self.processor).to(self.device)
        with torch.inference_mode():
            output = self.model(pixel_values)
        logit = _extract_output_logit(output)
        return float(logit.detach().to(device="cpu", dtype=torch.float32).item())


def resolve_torch_device(requested: str | torch.device | None) -> torch.device:
    """Use CUDA only after an explicit request and an availability check."""

    if requested is None:
        return torch.device("cpu")
    if isinstance(requested, torch.device):
        candidate = requested
    elif isinstance(requested, str) and requested.strip():
        normalized = requested.strip().lower()
        if normalized == "auto":
            return torch.device("cpu")
        try:
            candidate = torch.device(normalized)
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"unsupported Torch device: {requested!r}") from error
    else:
        raise ValueError("Torch device must be 'cpu', 'cuda', 'auto', or a torch.device")
    if candidate.type == "cuda":
        return candidate if torch.cuda.is_available() else torch.device("cpu")
    if candidate.type != "cpu":
        raise ValueError("only CPU or explicitly requested CUDA inference is supported")
    return candidate


def _extract_output_logit(output: Any) -> Tensor:
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
    if not isinstance(candidate, Tensor) or candidate.numel() != 1:
        raise DataIntegrityError("Torch model output must contain a single finite logit")
    detached = candidate.detach()
    if not detached.is_floating_point() or not torch.isfinite(detached).all().item():
        raise DataIntegrityError("Torch model output must contain a single finite logit")
    value = float(detached.reshape(()).to(device="cpu", dtype=torch.float32).item())
    if not math.isfinite(value):
        raise DataIntegrityError("Torch model output must contain a single finite logit")
    return detached.reshape(())


def _extract_model_state(payload: object) -> Mapping[str, Tensor]:
    if not isinstance(payload, Mapping):
        raise TrainingError("checkpoint must contain a model state mapping")
    for key in ("model_state_dict", "model_state", "model"):
        candidate = payload.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    if payload and all(isinstance(key, str) for key in payload) and all(
        isinstance(value, Tensor) for value in payload.values()
    ):
        return payload
    raise TrainingError("checkpoint does not contain a model state mapping")


def _checkpoint_model_version(payload: object) -> str | None:
    if not isinstance(payload, Mapping):
        return None
    value = payload.get("model_version")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _nonempty_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value
