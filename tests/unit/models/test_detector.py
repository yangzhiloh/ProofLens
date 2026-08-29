from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn


class TinyDino(nn.Module):
    def __init__(self, hidden_size: int = 8) -> None:
        super().__init__()
        self.projection = nn.Linear(3, hidden_size)
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for _ in range(4)]
        )

    def forward(self, *, pixel_values: torch.Tensor) -> SimpleNamespace:
        pooled = pixel_values.mean(dim=(-2, -1))
        hidden = self.projection(pooled)
        for layer in self.encoder.layer:
            hidden = torch.tanh(layer(hidden))
        tokens = torch.stack((hidden, hidden), dim=1)
        return SimpleNamespace(last_hidden_state=tokens)


def test_detector_returns_binary_logits_and_normalized_features() -> None:
    from prooflens.models.detector import DinoDetector

    model = DinoDetector(backbone=TinyDino(), hidden_size=8)

    output = model(torch.randn(2, 3, 28, 28))

    assert output.logits.shape == (2,)
    assert output.features.shape == (2, 8)
    assert torch.allclose(output.features.norm(dim=1), torch.ones(2), atol=1e-5)


def test_head_stage_freezes_backbone_and_trains_complete_head() -> None:
    from prooflens.models.detector import DinoDetector

    model = DinoDetector(backbone=TinyDino(), hidden_size=8)
    model.set_trainable_stage("head")

    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.feature_norm.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_last2_stage_unfreezes_only_final_two_blocks() -> None:
    from prooflens.models.detector import DinoDetector

    model = DinoDetector(backbone=TinyDino(), hidden_size=8)
    model.set_trainable_stage("last2")
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }

    assert any("backbone.encoder.layer.2" in name for name in trainable)
    assert any("backbone.encoder.layer.3" in name for name in trainable)
    assert not any("backbone.encoder.layer.1" in name for name in trainable)
    assert not any("backbone.projection" in name for name in trainable)
    assert any("classifier" in name for name in trainable)


def test_detector_rejects_unknown_stage_and_wrong_pixel_shape() -> None:
    from prooflens.models.detector import DinoDetector

    model = DinoDetector(backbone=TinyDino(), hidden_size=8)

    with pytest.raises(ValueError, match="stage"):
        model.set_trainable_stage("all")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=r"\[batch, 3, height, width\]"):
        model(torch.randn(2, 8))


def test_prediction_records_preprocessing_version() -> None:
    from prooflens.models.types import Prediction

    prediction = Prediction(
        probability_ai=0.75,
        probability_real=0.25,
        confidence=0.75,
        logit=1.1,
        model_version="fixture-v1",
        preprocessing_version="dinov2-base-224-v1",
        inference_ms=12.5,
    )

    assert prediction.preprocessing_version == "dinov2-base-224-v1"
