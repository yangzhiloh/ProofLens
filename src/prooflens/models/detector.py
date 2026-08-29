from __future__ import annotations

from typing import Literal

from torch import Tensor, nn
from torch.nn import functional

from prooflens.models.types import DetectorOutput

TrainableStage = Literal["head", "last2"]


class DinoDetector(nn.Module):
    """DINOv2 class-token detector with a small binary classification head."""

    def __init__(self, backbone: nn.Module, hidden_size: int = 768) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive")
        self.backbone = backbone
        self.feature_norm = nn.LayerNorm(hidden_size)
        self.classifier = nn.Linear(hidden_size, 1)

    @classmethod
    def from_pretrained(
        cls, model_id: str = "facebook/dinov2-base"
    ) -> DinoDetector:
        from transformers import Dinov2Model

        backbone = Dinov2Model.from_pretrained(model_id)
        hidden_size = int(backbone.config.hidden_size)
        return cls(backbone=backbone, hidden_size=hidden_size)

    def forward(self, pixel_values: Tensor) -> DetectorOutput:
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 3:
            raise ValueError("pixel_values must have shape [batch, 3, height, width]")
        hidden = self.backbone(pixel_values=pixel_values).last_hidden_state[:, 0]
        normalized = self.feature_norm(hidden)
        logits = self.classifier(normalized).squeeze(-1)
        return DetectorOutput(
            logits=logits,
            features=functional.normalize(normalized, dim=1),
        )

    def set_trainable_stage(self, stage: TrainableStage) -> None:
        if stage not in ("head", "last2"):
            raise ValueError(f"unsupported trainable stage: {stage!r}")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for parameter in self.feature_norm.parameters():
            parameter.requires_grad = True
        for parameter in self.classifier.parameters():
            parameter.requires_grad = True
        if stage == "last2":
            try:
                final_layers = self.backbone.encoder.layer[-2:]
            except (AttributeError, TypeError) as error:
                raise ValueError(
                    "backbone must expose encoder.layer for the last2 stage"
                ) from error
            if len(final_layers) < 2:
                raise ValueError("backbone must have at least two encoder layers")
            for layer in final_layers:
                for parameter in layer.parameters():
                    parameter.requires_grad = True
