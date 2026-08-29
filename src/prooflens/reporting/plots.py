"""Small report plots kept separate from metric calculation."""

from __future__ import annotations

from pathlib import Path

from prooflens.evaluation.metrics import MetricReport


def write_auc_plot(report: MetricReport, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Clean", *report.family_auc.keys()]
    values = [report.clean_auc, *report.family_auc.values()]
    figure, axis = plt.subplots(figsize=(9, 4.5))
    axis.bar(labels, values)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("ROC AUC")
    axis.tick_params(axis="x", rotation=30)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def write_condition_plot(report: MetricReport, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = list(report.condition_auc)
    values = list(report.condition_auc.values())
    figure, axis = plt.subplots(figsize=(10, 4.5))
    axis.bar(labels, values)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("ROC AUC")
    axis.tick_params(axis="x", rotation=60)
    figure.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path
