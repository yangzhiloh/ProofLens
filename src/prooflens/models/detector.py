"""DINOv2 detector wrapper with explicit frozen and fine-tuned stages."""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from prooflens.models.types import DetectorOutput


class DinoDetector(nn.Module):
    """A small binary head on top of a DINOv2 class-token representation."""

    def __init__(self, backbone: nn.Module, hidden_size: int = 768) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch.nn.Module")
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.backbone = backbone
        self.hidden_size = int(hidden_size)
        self.feature_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)
        self.model_version = "facebook/dinov2-base"

    @classmethod
    def from_pretrained(cls, model_name: str = "facebook/dinov2-base") -> "DinoDetector":
        from transformers import Dinov2Model

        backbone = Dinov2Model.from_pretrained(model_name)
        hidden_size = int(getattr(backbone.config, "hidden_size", 768))
        result = cls(backbone=backbone, hidden_size=hidden_size)
        result.model_version = model_name
        return result

    def forward(self, pixel_values: Tensor) -> DetectorOutput:
        if not isinstance(pixel_values, Tensor) or pixel_values.ndim != 4:
            raise ValueError("pixel_values must be a rank-4 torch tensor")
        output = self.backbone(pixel_values=pixel_values)
        hidden = _class_token(output)
        if hidden.ndim != 2 or hidden.shape[-1] != self.hidden_size:
            raise ValueError(
                "backbone class-token output must have shape [batch, hidden_size]"
            )
        normalized = self.feature_norm(hidden)
        logits = self.classifier(normalized).squeeze(-1)
        return DetectorOutput(logits=logits, features=F.normalize(normalized, dim=1))

    def set_trainable_stage(self, stage: Literal["head", "last2"]) -> None:
        if stage not in {"head", "last2"}:
            raise ValueError("stage must be 'head' or 'last2'")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in (*self.feature_norm.parameters(), *self.classifier.parameters()):
            parameter.requires_grad = True
        if stage == "last2":
            layers = getattr(getattr(self.backbone, "encoder", None), "layer", None)
            if layers is None or len(layers) < 2:
                raise ValueError("backbone does not expose at least two encoder layers")
            for block in layers[-2:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True


def _class_token(output: Any) -> Tensor:
    if isinstance(output, dict):
        hidden = output.get("last_hidden_state")
    else:
        hidden = getattr(output, "last_hidden_state", None)
    if not isinstance(hidden, Tensor):
        raise ValueError("backbone output must contain last_hidden_state")
    return hidden[:, 0]
