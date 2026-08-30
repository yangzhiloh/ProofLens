from pathlib import Path
import re
import tomllib


ROOT = Path(__file__).resolve().parents[2]


def _package_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip().lower()


def _project_metadata() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]


def test_original_core_dependencies_remain_declared() -> None:
    metadata = _project_metadata()
    dependencies = {_package_name(value) for value in metadata["dependencies"]}
    assert dependencies == {
        "torch", "torchvision", "transformers", "datasets", "huggingface-hub",
        "pillow", "numpy", "pandas", "pyarrow", "scikit-learn", "pydantic", "pyyaml",
        "imagehash", "matplotlib", "seaborn", "safetensors", "gradio", "onnx",
        "onnxscript", "onnxruntime",
    }


def test_lock_pins_every_direct_profile_dependency() -> None:
    metadata = _project_metadata()
    expected = {_package_name(value) for value in metadata["dependencies"]}
    for values in metadata["optional-dependencies"].values():
        expected.update(_package_name(value) for value in values)
    lock = (ROOT / "requirements.lock").read_text(encoding="utf-8").lower()
    for package in expected:
        assert re.search(rf"(?m)^{re.escape(package)}==[^\s]+$", lock)
