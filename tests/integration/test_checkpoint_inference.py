from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn


class TinyDetector(nn.Module):
    def __init__(self, bias: float = 0.0) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(bias))

    def forward(self, pixel_values: torch.Tensor):
        logits = pixel_values.mean(dim=(1, 2, 3)) + self.bias
        return SimpleNamespace(logits=logits)


class FakeProcessor:
    def __call__(self, *, images, return_tensors: str):
        assert return_tensors == "pt"
        values = [sum(image.getpixel((0, 0))) / (3 * 255) for image in images]
        return {
            "pixel_values": torch.stack(
                [torch.full((3, 224, 224), value, dtype=torch.float32) for value in values]
            )
        }


def test_role2_backend_loads_real_role1_checkpoint(tmp_path) -> None:
    from prooflens.inference.torch_backend import TorchLogitBackend
    from prooflens.training.checkpoints import CheckpointManager

    trained = TinyDetector(bias=1.5)
    optimizer = torch.optim.AdamW(trained.parameters(), lr=1e-3)
    loss = trained(torch.zeros(1, 3, 224, 224)).logits.sum()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        trained.bias.fill_(1.5)
    manager = CheckpointManager(tmp_path)
    checkpoint = manager.save(
        "selected",
        trained,
        optimizer,
        epoch=2,
        global_step=11,
        config_hash="config-hash",
    )
    best_checkpoint = manager.mark_best(checkpoint)

    backend = TorchLogitBackend.from_checkpoint(
        best_checkpoint,
        model_factory=TinyDetector,
        processor=FakeProcessor(),
    )

    assert backend.predict_logit(Image.new("RGB", (4, 4))) == pytest.approx(1.5)
