from __future__ import annotations

from pathlib import Path

import pytest


def test_openvino_compiles_onnx_when_optional_runtime_is_installed(tmp_path) -> None:
    pytest.importorskip("openvino")
    from prooflens.export.openvino_export import compile_openvino

    missing = tmp_path / "missing.onnx"
    with pytest.raises(FileNotFoundError):
        compile_openvino(missing)


def test_openvino_module_import_does_not_require_optional_runtime() -> None:
    from prooflens.export.openvino_export import OpenVinoSmokeReport

    report = OpenVinoSmokeReport(
        success=False,
        device="AUTO",
        first_prediction=None,
        error="runtime unavailable",
        onnx_path=Path("model.onnx"),
    )

    assert not report.success
    assert report.error == "runtime unavailable"
