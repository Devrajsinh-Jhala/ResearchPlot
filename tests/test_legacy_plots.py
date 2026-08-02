from __future__ import annotations

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage

import researchplot as rp
from researchplot import plots


@pytest.fixture(autouse=True)
def ignore_deprecations() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        yield


def test_top_level_legacy_exports_are_lazy_and_complete() -> None:
    assert rp.pairplot is plots.pairplot
    assert rp.stacked_bar is plots.stacked_bar
    assert "show" in inspect.signature(rp.bar).parameters
    assert "ax" in inspect.signature(rp.bar).parameters


def test_basic_helpers_accept_lists_and_return_artists() -> None:
    fig, ax = rp.bar([1, 2], ["a", "b"], show=False)
    assert len(ax.patches) == 2
    external_fig, external_ax = plt.subplots()
    returned_fig, returned_ax = rp.line([0, 1], [1, 2], "x", "y", ax=external_ax, show=False)
    assert (returned_fig, returned_ax) == (external_fig, external_ax)
    assert len(external_ax.lines) == 1

    _, ax = rp.stacked_bar([[1, 2], [2, 3]], ["A", "B"], show=False)
    assert len(ax.patches) == 4
    _, ax = rp.scatter([0, 1], [0, 2], "x", "y", show=False)
    assert len(ax.collections) == 1
    _, ax = rp.histogram([1, 2, 2, 3], show=False)
    assert len(ax.patches) == 10
    _, ax = rp.boxplot([[1, 2], [2, 3]], labels=["A", "B"], show=False)
    assert len(ax.lines) > 0
    _, ax = rp.pie([2, 3], ["A", "B"], show=False)
    assert len(ax.patches) == 2


def test_matrix_density_and_contour_helpers() -> None:
    matrix = np.array([[3, 1], [2, 4]])
    _, ax = rp.heatmap(matrix, ["A", "B"], ["C", "D"], show=False)
    assert len(ax.images) == 1
    _, ax = rp.confusion_matrix(matrix, ["A", "B"], show=False)
    assert len(ax.texts) == 4
    x, y = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    _, ax = rp.contour_plot(x, y, x**2 + y**2, show=False)
    assert ax.collections
    _, ax = rp.hexbin([0, 1, 1], [0, 1, 2], show=False)
    assert ax.collections


def test_training_metric_and_series_helpers() -> None:
    epochs = np.arange(3)
    _, ax = rp.accuracy_vs_epoch(epochs, [0.5, 0.7, 0.8], np.array([0.4, 0.6, 0.75]), show=False)
    assert len(ax.lines) == 2
    _, ax = rp.loss_vs_epoch(epochs, [2, 1, 0.5], np.array([2.2, 1.2, 0.7]), show=False)
    assert len(ax.lines) == 2
    _, ax = rp.roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], show=False)
    assert "AUC" in ax.get_legend().get_texts()[0].get_text()
    _, ax = rp.precision_recall_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], show=False)
    assert len(ax.lines) == 1
    _, ax = rp.learning_curves([1, 2, 3], [0.5, 0.7, 0.8], [0.4, 0.6, 0.7], show=False)
    assert len(ax.lines) == 2
    _, ax = rp.time_series([1, 2, 3], [3, 2, 4], show=False)
    assert len(ax.lines) == 1


def test_specialized_helpers() -> None:
    _, ax = rp.violinplot([[1, 2, 3], [2, 3, 4]], labels=["A", "B"], show=False)
    assert ax.collections
    frame = pd.DataFrame({"a": [1, 2, 3], "b": [3, 2, 1], "group": ["x", "x", "y"]})
    grid = rp.pairplot(frame, ["a", "b"], hue="group", show=False)
    assert grid.fig is not None
    _, ax = rp.radar_chart(["A", "B", "C"], [1, 2, 3], show=False)
    assert ax.name == "polar"
    matrix = linkage(np.array([[0], [1], [3], [10]]), method="single")
    _, ax = rp.dendrogram(matrix, show=False)
    assert ax.collections
    x, y = np.meshgrid([0, 1], [0, 1])
    _, ax = rp.quiver(x, y, np.ones_like(x), np.ones_like(y), show=False)
    assert ax.collections
    _, ax = rp.surface_3d(x, y, x + y, show=False)
    assert ax.name == "3d"
    _, ax = rp.sankey([-1, 1], ["in", "out"], show=False)
    assert ax.patches
    _, ax = rp.error_band([0, 1], [1, 2], [0.1, 0.2], show=False)
    assert len(ax.lines) == 1 and ax.collections


def test_confidence_interval_is_rendered_once() -> None:
    samples = [[0.8, 1.8], [1.0, 2.0], [1.2, 2.2]]
    _, ax = rp.line(
        [0, 1], [1, 2], "x", "y", show_confidence_interval=True, ci_data=samples, show=False
    )
    assert len(ax.lines) == 1
    assert len(ax.collections) == 1


def test_intentional_errors_and_unknown_format_warning() -> None:
    with pytest.raises(ValueError, match="same length"):
        rp.bar([1, 2], ["only-one"], show=False)
    with pytest.raises(ValueError, match="ci_data"):
        rp.line([0], [1], "x", "y", show_confidence_interval=True, show=False)
    with pytest.raises(ValueError, match="positive total"):
        rp.pie([0, 0], ["A", "B"], show=False)
    with pytest.warns(DeprecationWarning, match="falling back"):
        rp.bar([1], format="unknown-conference", show=False)
