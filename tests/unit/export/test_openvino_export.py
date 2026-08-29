import pytest


def test_openvino_compile_is_optional() -> None:
    pytest.importorskip("openvino")
    from prooflens.export.openvino_export import compile_openvino

    assert callable(compile_openvino)
