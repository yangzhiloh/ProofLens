from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from PIL import Image


class _PixelService:
    def predict(self, image: Image.Image) -> SimpleNamespace:
        red = image.convert("RGB").getpixel((0, 0))[0]
        return SimpleNamespace(probability_ai=red / 255.0)


def test_directory_predictions_are_recursive_sorted_and_portable(tmp_path) -> None:
    from prooflens.inference.directory import write_directory_predictions

    input_dir = tmp_path / "images"
    nested = input_dir / "nested"
    nested.mkdir(parents=True)
    Image.new("RGB", (4, 4), (51, 0, 0)).save(input_dir / "b.jpg")
    Image.new("RGB", (4, 4), (204, 0, 0)).save(input_dir / "a.png")
    Image.new("RGB", (4, 4), (102, 0, 0)).save(nested / "c.webp", lossless=True)
    (input_dir / "ignore.txt").write_text("not an image", encoding="utf-8")
    output = tmp_path / "predictions.json"

    result = write_directory_predictions(input_dir, output, _PixelService())

    assert result == output
    assert json.loads(output.read_text(encoding="utf-8")) == [
        {"image_path": "a.png", "pred": pytest.approx(0.8)},
        {"image_path": "b.jpg", "pred": pytest.approx(0.2, abs=0.02)},
        {"image_path": "nested/c.webp", "pred": pytest.approx(0.4)},
    ]


def test_directory_predictions_reject_empty_input_without_writing(tmp_path) -> None:
    from prooflens.errors import UserInputError
    from prooflens.inference.directory import write_directory_predictions

    input_dir = tmp_path / "empty"
    input_dir.mkdir()
    output = tmp_path / "predictions.json"

    with pytest.raises(UserInputError, match="no supported images"):
        write_directory_predictions(input_dir, output, _PixelService())

    assert not output.exists()


def test_directory_predictions_identify_corrupt_image_and_preserve_output(tmp_path) -> None:
    from prooflens.errors import UserInputError
    from prooflens.inference.directory import write_directory_predictions

    input_dir = tmp_path / "images"
    input_dir.mkdir()
    (input_dir / "broken.png").write_bytes(b"not a png")
    output = tmp_path / "predictions.json"
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(UserInputError, match="broken.png"):
        write_directory_predictions(input_dir, output, _PixelService())

    assert output.read_text(encoding="utf-8") == "existing\n"
