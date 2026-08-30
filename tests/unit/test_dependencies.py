import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _package_name(requirement: str) -> str:
    return re.split(r"[<>=!~;\[]", requirement, maxsplit=1)[0].strip().lower()


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def _project_metadata() -> dict:
    return _pyproject()["project"]


def _lock() -> dict:
    with (ROOT / "uv.lock").open("rb") as stream:
        return tomllib.load(stream)


def test_original_core_dependencies_remain_declared() -> None:
    metadata = _project_metadata()
    dependencies = {_package_name(value) for value in metadata["dependencies"]}
    assert dependencies == {
        "torch", "torchvision", "transformers", "datasets", "huggingface-hub",
        "pillow", "numpy", "pandas", "pyarrow", "scikit-learn", "pydantic", "pyyaml",
        "imagehash", "matplotlib", "seaborn", "safetensors", "gradio", "onnx",
        "onnxscript", "onnxruntime",
    }


def test_lock_covers_every_direct_profile_dependency() -> None:
    metadata = _project_metadata()
    expected = {_package_name(value) for value in metadata["dependencies"]}
    for values in metadata["optional-dependencies"].values():
        expected.update(_package_name(value) for value in values)
    locked = {package["name"].lower() for package in _lock()["package"]}
    assert expected <= locked


def test_reviewed_direct_versions_are_exact_uv_constraints() -> None:
    metadata = _project_metadata()
    expected = {_package_name(value) for value in metadata["dependencies"]}
    for values in metadata["optional-dependencies"].values():
        expected.update(_package_name(value) for value in values)
    constraints = _pyproject()["tool"]["uv"]["constraint-dependencies"]
    assert {_package_name(value) for value in constraints} == expected
    assert all(re.fullmatch(r"[^=<>!~;\[]+==[^\s]+", value) for value in constraints)


def test_lock_covers_the_complete_supported_python_range() -> None:
    lock = _lock()
    assert lock["requires-python"] == ">=3.11, <3.13"
    assert set(lock["resolution-markers"]) == {
        "python_full_version >= '3.12'",
        "python_full_version < '3.12'",
    }


def test_every_registry_artifact_has_a_sha256_hash() -> None:
    registry_packages = [
        package for package in _lock()["package"] if "registry" in package.get("source", {})
    ]
    assert len(registry_packages) > 24
    for package in registry_packages:
        artifacts = ([package["sdist"]] if "sdist" in package else []) + package.get("wheels", [])
        assert artifacts, package["name"]
        for artifact in artifacts:
            assert re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["hash"]), package["name"]
