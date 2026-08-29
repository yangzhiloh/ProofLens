import torch
from torch import nn

from prooflens.models.detector import DinoDetector


class TinyDino(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.layer = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])

    def forward(self, pixel_values):
        hidden = pixel_values.mean(dim=(2, 3)).unsqueeze(1).repeat(1, 2, 1)
        return type("Output", (), {"last_hidden_state": hidden})()


def test_detector_returns_binary_logits_and_normalized_features() -> None:
    model = DinoDetector(TinyDino(), hidden_size=3)
    output = model(torch.randn(2, 3, 28, 28))
    assert output.logits.shape == (2,)
    assert output.features.shape == (2, 3)
    assert torch.allclose(output.features.norm(dim=1), torch.ones(2), atol=1e-5)


def test_last2_stage_unfreezes_only_final_two_blocks() -> None:
    model = DinoDetector(TinyDino(), hidden_size=3)
    model.set_trainable_stage("last2")
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert any("encoder.layer.2" in name for name in trainable)
    assert any("encoder.layer.3" in name for name in trainable)
    assert not any("encoder.layer.1" in name for name in trainable)
    assert any("classifier" in name for name in trainable)
