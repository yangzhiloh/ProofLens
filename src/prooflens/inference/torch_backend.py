"""PyTorch checkpoint backend with an explicit CPU fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch
from PIL import Image

from prooflens.inference.preprocess import ImageProcessor, create_dinov2_processor, preprocess_images
from prooflens.models.detector import DinoDetector


class TorchLogitBackend:
    def __init__(
        self,
        checkpoint: Path | str | torch.nn.Module,
        processor: ImageProcessor | None = None,
        model_version: str = "prooflens-torch",
        device: str | torch.device | None = None,
        model: torch.nn.Module | None = None,
    ) -> None:
        self.device = _resolve_device(device)
        self.processor = processor or create_dinov2_processor()
        self.model = model or _load_model(checkpoint)
        self.model.to(self.device).eval()
        self.model_version = model_version

    def predict_logit(self, image: Image.Image) -> float:
        pixels = preprocess_images([image], processor=self.processor).to(self.device)
        with torch.no_grad():
            output = self.model(pixels)
            logits = output.logits if hasattr(output, "logits") else output
        return float(torch.as_tensor(logits).reshape(-1)[0].detach().cpu())


TorchBackend = TorchLogitBackend


def _load_model(checkpoint: Path | str | torch.nn.Module) -> torch.nn.Module:
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint
    path = Path(checkpoint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, torch.nn.Module):
        return payload
    if isinstance(payload, Mapping) and isinstance(payload.get("model_object"), torch.nn.Module):
        return payload["model_object"]
    model = DinoDetector.from_pretrained("facebook/dinov2-base")
    state = payload.get("model", payload) if isinstance(payload, Mapping) else payload
    model.load_state_dict(state)
    return model


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        # Production inference is deliberately CPU-first. CUDA is used only
        # when the caller explicitly requests it and it is available.
        return torch.device("cpu")
    requested = torch.device(device)
    return torch.device("cpu") if requested.type == "cuda" and not torch.cuda.is_available() else requested
