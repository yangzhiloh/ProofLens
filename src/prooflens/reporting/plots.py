from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from prooflens.evaluation.metrics import MetricReport
from prooflens.reporting.tables import FAMILY_DISPLAY


def write_auc_plot(report: MetricReport, path: Path) -> Path:
    """Write a headless clean-versus-family ROC AUC bar chart."""

    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("AUC plot path must use the .png suffix")
    labels = ["Clean", *(FAMILY_DISPLAY[family] for family in FAMILY_DISPLAY)]
    values = [report.clean_auc, *(report.family_auc[family] for family in FAMILY_DISPLAY)]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    try:
        axis.bar(labels, values, color="#315f8c")
        axis.set_ylim(0.0, 1.0)
        axis.set_ylabel("ROC AUC")
        axis.tick_params(axis="x", rotation=30)
        figure.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.stem}.{uuid4().hex}.png")
        try:
            figure.savefig(temporary_path, dpi=160, format="png")
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    finally:
        plt.close(figure)
    return path
