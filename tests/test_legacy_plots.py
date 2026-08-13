from __future__ import annotations

import inspect
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage

import researchplot as rp
from researchplot import plots
from researchplot.config import LegacyStyleWarning


@pytest.fixture(autouse=True)
def ignore_legacy_deprecations() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        yield
    plt.close("all")


def test_legacy_exports_are_complete_and_preserve_composition_keywords() -> None:
    assert rp.bar is plots.bar
    assert rp.pairplot is plots.pairplot
    assert "bar" in dir(rp)
    assert plots.pairplot is not None
    assert plots.stacked_bar is not None
    assert "show" in inspect.signature(plots.bar).parameters
    assert "ax" in inspect.signature(plots.bar).parameters
    assert set(plots.__all__) == {
        "PlotStyle",
        "accuracy_vs_epoch",
        "bar",
        "boxplot",
        "confusion_matrix",
        "contour_plot",
        "dendrogram",
        "error_band",
        "heatmap",
        "hexbin",
        "histogram",
        "learning_curves",
        "line",
        "loss_vs_epoch",
        "pairplot",
        "pie",
        "precision_recall_curve",
        "quiver",
        "radar_chart",
        "roc_curve",
        "sankey",
        "scatter",
        "stacked_bar",
        "surface_3d",
        "time_series",
        "violinplot",
    }


def test_importing_compatibility_module_does_not_eagerly_import_optional_integrations() -> None:
    code = (
        "import sys; import researchplot.plots; "
        "blocked={'pandas','seaborn','sklearn','scipy'}; "
        "loaded={name.split('.')[0] for name in sys.modules}; "
        "assert not blocked & loaded, blocked & loaded"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_basic_helpers_accept_lists_return_artists_and_save(tmp_path: Path) -> None:
    output = tmp_path / "bar.png"
    fig, ax = plots.bar([1, 2], ["a", "b"], output_path=output, show=False)
    assert fig is ax.figure
    assert len(ax.patches) == 2
    assert output.is_file()

    external_fig, external_ax = plt.subplots()
    returned_fig, returned_ax = plots.line([0, 1], [1, 2], "x", "y", ax=external_ax, show=False)
    assert (returned_fig, returned_ax) == (external_fig, external_ax)
    assert len(external_ax.lines) == 1

    _, ax = plots.stacked_bar([[1, 2], [2, 3]], ["A", "B"], show=False)
    assert len(ax.patches) == 4
    _, ax = plots.scatter([0, 1], [0, 2], "x", "y", show=False)
    assert len(ax.collections) == 1
    _, ax = plots.histogram([1, 2, 2, 3], show=False)
    assert len(ax.patches) == 10
    _, ax = plots.boxplot([[1, 2], [2, 3]], labels=["A", "B"], show=False)
    assert len(ax.lines) > 0
    _, ax = plots.pie([2, 3], ["A", "B"], show=False)
    assert len(ax.patches) == 2


def test_matrix_density_and_contour_helpers() -> None:
    matrix = np.array([[3, 1], [2, 4]])
    _, ax = plots.heatmap(matrix, ["A", "B"], ["C", "D"], show=False)
    assert len(ax.images) == 1
    _, ax = plots.confusion_matrix(matrix, ["A", "B"], show=False)
    assert len(ax.texts) == 4
    x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    _, ax = plots.contour_plot(x, y, x**2 + y**2, show=False)
    assert ax.collections
    _, ax = plots.hexbin([0, 1, 1], [0, 1, 2], show=False)
    assert ax.collections


def test_training_metric_and_series_helpers() -> None:
    epochs = np.arange(3)
    _, ax = plots.accuracy_vs_epoch(epochs, [0.5, 0.7, 0.8], np.array([0.4, 0.6, 0.75]), show=False)
    assert len(ax.lines) == 2
    _, ax = plots.loss_vs_epoch(epochs, [2, 1, 0.5], np.array([2.2, 1.2, 0.7]), show=False)
    assert len(ax.lines) == 2
    _, ax = plots.roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], show=False)
    assert "AUC" in ax.get_legend().get_texts()[0].get_text()
    _, ax = plots.precision_recall_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], show=False)
    assert len(ax.lines) == 1
    _, ax = plots.learning_curves([1, 2, 3], [0.5, 0.7, 0.8], [0.4, 0.6, 0.7], show=False)
    assert len(ax.lines) == 2
    _, ax = plots.time_series([1, 2, 3], [3, 2, 4], show=False)
    assert len(ax.lines) == 1


def test_specialized_helpers() -> None:
    _, ax = plots.violinplot([[1, 2, 3], [2, 3, 4]], labels=["A", "B"], show=False)
    assert ax.collections
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1], "group": ["x", "x", "y"]})
    grid = plots.pairplot(frame, ["a", "b"], hue="group", show=False)
    assert grid.fig is not None
    _, ax = plots.radar_chart(["A", "B", "C"], [1, 2, 3], show=False)
    assert ax.name == "polar"
    matrix = linkage(np.array([[0], [1], [3], [10]]), method="single")
    _, ax = plots.dendrogram(matrix, show=False)
    assert ax.collections
    x, y = np.meshgrid([0, 1], [0, 1])
    _, ax = plots.quiver(x, y, np.ones_like(x), np.ones_like(y), show=False)
    assert ax.collections
    _, ax = plots.surface_3d(x, y, x + y, show=False)
    assert ax.name == "3d"
    _, ax = plots.sankey([-1, 1], ["in", "out"], show=False)
    assert ax.patches
    _, ax = plots.error_band([0, 1], [1, 2], [0.1, 0.2], show=False)
    assert len(ax.lines) == 1 and ax.collections


def test_confidence_interval_is_rendered_once() -> None:
    samples = [[0.8, 1.8], [1.0, 2.0], [1.2, 2.2]]
    _, ax = plots.line(
        [0, 1],
        [1, 2],
        "x",
        "y",
        show_confidence_interval=True,
        ci_data=samples,
        show=False,
    )
    assert len(ax.lines) == 1
    assert len(ax.collections) == 1


def test_intentional_errors_and_legacy_format_warnings() -> None:
    with pytest.raises(ValueError, match="same length"):
        plots.bar([1, 2], ["only-one"], show=False)
    with pytest.raises(ValueError, match="ci_data"):
        plots.line([0], [1], "x", "y", show_confidence_interval=True, show=False)
    with pytest.raises(ValueError, match="positive total"):
        plots.pie([0, 0], ["A", "B"], show=False)
    with pytest.warns(DeprecationWarning, match="falling back"):
        plots.bar([1], format="unknown-conference", show=False)
    with pytest.warns(LegacyStyleWarning, match="unverified"):
        plots.PlotStyle("science")


def test_missing_optional_dependency_error_is_targeted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = plots.import_module

    def fail_seaborn(name: str) -> object:
        if name == "seaborn":
            raise ModuleNotFoundError("No module named 'seaborn'", name="seaborn")
        return real_import(name)

    monkeypatch.setattr(plots, "import_module", fail_seaborn)
    with pytest.raises(ImportError, match=r"researchplot-venues\[plots\]"):
        plots.violinplot([[1, 2]], show=False)
