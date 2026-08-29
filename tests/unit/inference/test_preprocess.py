from __future__ import annotations

import builtins
import importlib
from types import SimpleNamespace

import torch
from PIL import Image


class FakeProcessor:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Image.Image, ...], str]] = []

    def __call__(self, *, images, return_tensors: str):
        copied = tuple(image.copy() for image in images)
        self.calls.append((copied, return_tensors))
        return {
            "pixel_values": torch.ones(
                (len(copied), 3, 224, 224), dtype=torch.float64
            )
        }


def test_shared_preprocessing_uses_injected_processor_without_duplicate_normalization() -> None:
    preprocess = importlib.import_module("prooflens.inference.preprocess")
    processor = FakeProcessor()
    images = [Image.new("RGB", (5, 4), "red"), Image.new("RGB", (3, 2), "blue")]

    pixels = preprocess.preprocess_images(images, processor=processor)

    assert pixels.shape == (2, 3, 224, 224)
    assert pixels.dtype == torch.float32
    assert torch.equal(pixels, torch.ones_like(pixels))
    assert len(processor.calls) == 1
    assert processor.calls[0][1] == "pt"
    assert [image.size for image in processor.calls[0][0]] == [(5, 4), (3, 2)]
    assert preprocess.PREPROCESSING_VERSION == "dinov2-base-224-v1"


def test_production_processor_import_and_pretrained_load_are_explicit_and_lazy(
    monkeypatch,
) -> None:
    real_import = builtins.__import__
    import_calls: list[str] = []
    pretrained_calls: list[str] = []

    class FakeAutoImageProcessor:
        @classmethod
        def from_pretrained(cls, model_id: str):
            pretrained_calls.append(model_id)
            return object()

    fake_transformers = SimpleNamespace(AutoImageProcessor=FakeAutoImageProcessor)

    def controlled_import(name: str, *args, **kwargs):
        if name == "transformers":
            import_calls.append(name)
            return fake_transformers
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", controlled_import)
    preprocess = importlib.import_module("prooflens.inference.preprocess")
    importlib.reload(preprocess)

    assert import_calls == []
    assert pretrained_calls == []

    processor = preprocess.create_dinov2_processor()

    assert processor is not None
    assert import_calls == ["transformers"]
    assert pretrained_calls == ["facebook/dinov2-base"]
