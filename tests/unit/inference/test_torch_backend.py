import numpy as np
import torch
from PIL import Image
from torch import nn

from prooflens.inference.torch_backend import TorchLogitBackend


class Processor:
    def __call__(self, *, images, return_tensors):
        values = [np.asarray(image.resize((224, 224)), dtype=np.float32).transpose(2, 0, 1) / 255.0 for image in images]
        return {"pixel_values": torch.tensor(np.stack(values))}


class ToyModel(nn.Module):
    def forward(self, pixel_values):
        return pixel_values.mean(dim=(1, 2, 3))


def test_torch_backend_uses_cpu_and_returns_logit() -> None:
    backend = TorchLogitBackend(ToyModel(), processor=Processor(), device="cuda")
    assert backend.device.type == "cpu"
    assert isinstance(backend.predict_logit(Image.new("RGB", (8, 8), color=(255, 255, 255))), float)
