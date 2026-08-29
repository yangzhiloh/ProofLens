from pathlib import Path

from prooflens.evaluation.metrics import MetricReport
from prooflens.reporting.plots import write_auc_plot
from prooflens.reporting.tables import write_robustness_markdown


def _report() -> MetricReport:
    families = {name: 0.8 for name in ("jpeg", "blur", "resize", "noise", "color_jitter", "center_crop")}
    conditions = {f"{name}_condition": 0.8 for name in families}
    return MetricReport(0.9, conditions, families, 0.8, 0.8, 0.8, 0.8, 0.75, 0.85)


def test_markdown_table_contains_required_rows(tmp_path: Path) -> None:
    path = write_robustness_markdown(_report(), tmp_path / "robustness.md")
    text = path.read_text(encoding="utf-8")
    for name in ("Clean", "JPEG", "Blur", "Resize", "Noise", "Color jitter", "Center crop"):
        assert name in text


def test_auc_plot_is_written(tmp_path: Path) -> None:
    path = write_auc_plot(_report(), tmp_path / "auc.png")
    assert path.is_file()
    assert path.stat().st_size > 0
