"""Deprecated plotting helpers retained as a lazy compatibility layer.

New code should style ordinary Matplotlib calls with :func:`researchplot.use`.
"""

from __future__ import annotations

import warnings
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.sankey import Sankey

from .config import PLOT_FORMATS
from .models import LegacyStyleWarning

_VERIFIED_LEGACY = {"ieee", "nature", "elsevier"}


def _require(module: str, feature: str) -> Any:
    try:
        return import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"{feature} requires optional plotting dependencies. "
            "Install them with: pip install 'researchplot[plots]'"
        ) from exc


class PlotStyle:
    """Compatibility object for the original mutable plot settings."""

    def __init__(self, format: str = "ieee") -> None:
        key = str(format).casefold().strip()
        if key not in PLOT_FORMATS:
            warnings.warn(
                f"Unknown legacy format {format!r}; falling back to 'ieee'. Silent fallback "
                "is deprecated. Use researchplot.resolve_venue() for strict resolution.",
                DeprecationWarning,
                stacklevel=2,
            )
            key = "ieee"
        if key not in _VERIFIED_LEGACY and key != "default":
            warnings.warn(
                f"Legacy style {key!r} is unverified and is not a compliance profile.",
                LegacyStyleWarning,
                stacklevel=2,
            )
        self.format = key
        self.params = PLOT_FORMATS[key].copy()

    def update(self, **kwargs: Any) -> PlotStyle:
        self.params.update(kwargs)
        return self

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def _configure_latex(self) -> None:
        if self.params.get("use_latex", False):
            plt.rcParams.update({"text.usetex": True, "font.family": "serif"})


def _prepare(
    format: str,
    overrides: dict[str, Any],
    ax: Axes | None,
    *,
    projection: str | None = None,
) -> tuple[Figure, Axes, dict[str, Any]]:
    warnings.warn(
        "researchplot.plots helpers are deprecated; use researchplot.use() with native "
        "Matplotlib instead.",
        FutureWarning,
        stacklevel=3,
    )
    params = PlotStyle(format).update(**overrides).params
    if ax is not None:
        return cast(Figure, ax.figure), ax, params
    if projection is None:
        fig, created_ax = plt.subplots(figsize=params["figsize"], dpi=params["dpi"])
    else:
        fig = plt.figure(figsize=params["figsize"], dpi=params["dpi"])
        created_ax = fig.add_subplot(111, projection=projection)
    return fig, created_ax, params


def _finish(
    fig: Figure,
    ax: Axes,
    output_path: str | Path | None,
    show: bool,
    dpi: int,
    *,
    tight: bool = False,
) -> tuple[Figure, Axes]:
    if output_path is not None:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight" if tight else None)
    if show:
        plt.show()
    return fig, ax


def _same_length(left: Any, right: Any, left_name: str, right_name: str) -> None:
    try:
        if len(left) != len(right):
            raise ValueError(f"{left_name} and {right_name} must have the same length.")
    except TypeError as exc:
        raise TypeError(f"{left_name} and {right_name} must be array-like inputs.") from exc


def bar(
    data: Any,
    xticks: Any = None,
    xlabel: str = "",
    ylabel: str = "",
    format: str = "ieee",
    title: str | None = None,
    output_path: str | Path | None = None,
    yerr: Any = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a bar plot and return its figure and axes."""

    fig, ax, params = _prepare(format, kwargs, ax)
    ax.bar(range(len(data)), data, yerr=yerr, linewidth=params["linewidth"])
    if xticks is not None:
        _same_length(data, xticks, "data", "xticks")
        ax.set_xticks(range(len(data)), xticks, fontsize=params["xtick_font_size"])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return _finish(fig, ax, output_path, show, params["dpi"])


def stacked_bar(
    data: Any,
    stack_labels: Any,
    xticks: Any = None,
    xlabel: str = "",
    ylabel: str = "",
    format: str = "ieee",
    title: str | None = None,
    output_path: str | Path | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a stacked bar plot."""

    _same_length(data, stack_labels, "data", "stack_labels")
    if len(data) == 0:
        raise ValueError("data must contain at least one stack.")
    fig, ax, params = _prepare(format, kwargs, ax)
    bottom = [0.0] * len(data[0])
    for stack, label in zip(data, stack_labels, strict=True):
        _same_length(stack, bottom, "stack", "first stack")
        ax.bar(range(len(stack)), stack, bottom=bottom, label=label, linewidth=params["linewidth"])
        bottom = [
            float(current) + float(value) for current, value in zip(bottom, stack, strict=True)
        ]
    if xticks is not None:
        _same_length(bottom, xticks, "bar groups", "xticks")
        ax.set_xticks(range(len(bottom)), xticks, fontsize=params["xtick_font_size"])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.legend(fontsize=params["legend_font_size"])
    return _finish(fig, ax, output_path, show, params["dpi"])


def scatter(
    x: Any,
    y: Any,
    xlabel: str,
    ylabel: str,
    format: str = "ieee",
    output_path: str | Path | None = None,
    show_correlation: bool = True,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a scatter plot, optionally annotating Pearson correlation."""

    _same_length(x, y, "x", "y")
    fig, ax, params = _prepare(format, kwargs, ax)
    ax.scatter(x, y, s=params["marker_size"], linewidths=params["linewidth"])
    if show_correlation:
        np = _require("numpy", "scatter correlation")
        corr = float(np.corrcoef(x, y)[0, 1])
        ax.text(0.05, 0.95, f"r = {corr:.2f}", transform=ax.transAxes, va="top")
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return _finish(fig, ax, output_path, show, params["dpi"])


def line(
    x: Any,
    y: Any,
    xlabel: str,
    ylabel: str,
    format: str = "ieee",
    output_path: str | Path | None = None,
    show_confidence_interval: bool = False,
    ci_data: Any = None,
    ci: float = 0.95,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a line plot with an optional percentile confidence interval."""

    _same_length(x, y, "x", "y")
    if show_confidence_interval and ci_data is None:
        raise ValueError("ci_data is required when show_confidence_interval=True.")
    if not 0 < ci < 1:
        raise ValueError("ci must be between 0 and 1.")
    fig, ax, params = _prepare(format, kwargs, ax)
    ax.plot(x, y, linewidth=params["linewidth"])
    if show_confidence_interval:
        np = _require("numpy", "confidence intervals")
        values = np.asarray(ci_data)
        if values.ndim != 2 or values.shape[1] != len(x):
            raise ValueError("ci_data must be a 2D array with one column per x value.")
        lower = np.percentile(values, (1 - ci) / 2 * 100, axis=0)
        upper = np.percentile(values, (1 + ci) / 2 * 100, axis=0)
        ax.fill_between(x, lower, upper, alpha=0.2)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return _finish(fig, ax, output_path, show, params["dpi"])


def histogram(
    data: Any,
    bins: Any = None,
    xlabel: str = "",
    ylabel: str = "",
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a histogram using the style's default bin count when omitted."""

    fig, ax, params = _prepare(format, kwargs, ax)
    ax.hist(data, bins=params["hist_bins"] if bins is None else bins, linewidth=params["linewidth"])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return _finish(fig, ax, output_path, show, params["dpi"])


def boxplot(
    data: Any,
    labels: Any = None,
    xlabel: str = "",
    ylabel: str = "",
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a box-and-whisker plot."""

    fig, ax, params = _prepare(format, kwargs, ax)
    ax.boxplot(data, widths=params["box_width"])
    if labels is not None:
        ax.set_xticks(range(1, len(labels) + 1), labels)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return _finish(fig, ax, output_path, show, params["dpi"])


def heatmap(
    data: Any,
    xticklabels: Any = None,
    yticklabels: Any = None,
    xlabel: str = "",
    ylabel: str = "",
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a matrix heatmap."""

    fig, ax, params = _prepare(format, kwargs, ax)
    image = ax.imshow(data, cmap=params["heatmap_cmap"], aspect="auto")
    fig.colorbar(image, ax=ax)
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)), xticklabels)
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels)), yticklabels)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    return _finish(fig, ax, output_path, show, params["dpi"])


def confusion_matrix(
    cm: Any,
    classes: Any,
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create an annotated confusion-matrix heatmap."""

    np = _require("numpy", "confusion_matrix")
    values = np.asarray(cm)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("cm must be a square two-dimensional array.")
    if len(classes) != values.shape[0]:
        raise ValueError("classes must contain one label per matrix row.")
    fig, ax, params = _prepare(format, kwargs, ax)
    image = ax.imshow(values, cmap=params["confusion_matrix_cmap"])
    fig.colorbar(image, ax=ax)
    threshold = float(values.max()) / 2 if values.size else 0
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                str(values[row, column]),
                ha="center",
                va="center",
                color="white" if values[row, column] > threshold else "black",
            )
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set(xlabel="Predicted Labels", ylabel="True Labels", title=title)
    return _finish(fig, ax, output_path, show, params["dpi"])


def accuracy_vs_epoch(
    epochs: Any,
    accuracy: Any,
    val_accuracy: Any = None,
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot training and optional validation accuracy by epoch."""

    _same_length(epochs, accuracy, "epochs", "accuracy")
    fig, ax, params = _prepare(format, kwargs, ax)
    ax.plot(
        epochs, accuracy, linewidth=params["accuracy_loss_linewidth"], label="Training Accuracy"
    )
    if val_accuracy is not None:
        _same_length(epochs, val_accuracy, "epochs", "val_accuracy")
        ax.plot(
            epochs,
            val_accuracy,
            linewidth=params["accuracy_loss_linewidth"],
            label="Validation Accuracy",
        )
    ax.set(xlabel="Epochs", ylabel="Accuracy", title=title)
    ax.legend(fontsize=params["legend_font_size"])
    return _finish(fig, ax, output_path, show, params["dpi"])


def loss_vs_epoch(
    epochs: Any,
    loss: Any,
    val_loss: Any = None,
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot training and optional validation loss by epoch."""

    _same_length(epochs, loss, "epochs", "loss")
    fig, ax, params = _prepare(format, kwargs, ax)
    ax.plot(epochs, loss, linewidth=params["accuracy_loss_linewidth"], label="Training Loss")
    if val_loss is not None:
        _same_length(epochs, val_loss, "epochs", "val_loss")
        ax.plot(
            epochs, val_loss, linewidth=params["accuracy_loss_linewidth"], label="Validation Loss"
        )
    ax.set(xlabel="Epochs", ylabel="Loss", title=title)
    ax.legend(fontsize=params["legend_font_size"])
    return _finish(fig, ax, output_path, show, params["dpi"])


def roc_curve(
    y_true: Any,
    y_scores: Any,
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot the receiver-operating characteristic and AUC."""

    metrics = _require("sklearn.metrics", "roc_curve")
    fpr, tpr, _ = metrics.roc_curve(y_true, y_scores)
    score = metrics.auc(fpr, tpr)
    fig, ax, params = _prepare(format, kwargs, ax)
    ax.plot(fpr, tpr, linewidth=params["linewidth"], label=f"AUC = {score:.2f}")
    ax.set(xlabel="False Positive Rate", ylabel="True Positive Rate", title=title)
    ax.legend(fontsize=params["legend_font_size"])
    return _finish(fig, ax, output_path, show, params["dpi"])


def precision_recall_curve(
    y_true: Any,
    y_scores: Any,
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot a precision-recall curve and AUC."""

    metrics = _require("sklearn.metrics", "precision_recall_curve")
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_scores)
    score = metrics.auc(recall, precision)
    fig, ax, params = _prepare(format, kwargs, ax)
    ax.plot(recall, precision, linewidth=params["linewidth"], label=f"AUC = {score:.2f}")
    ax.set(xlabel="Recall", ylabel="Precision", title=title)
    ax.legend(fontsize=params["legend_font_size"])
    return _finish(fig, ax, output_path, show, params["dpi"])


def violinplot(
    data: Any,
    labels: Any = None,
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a Seaborn violin plot."""

    sns = _require("seaborn", "violinplot")
    fig, ax, params = _prepare(format, kwargs, ax)
    sns.violinplot(data=data, linewidth=params["linewidth"], ax=ax)
    if labels is not None:
        ax.set_xticks(range(len(labels)), labels)
    ax.set(xlabel="Groups", ylabel="Values", title=title)
    return _finish(fig, ax, output_path, show, params["dpi"])


def contour_plot(
    X: Any,
    Y: Any,
    Z: Any,
    levels: Any = None,
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a filled contour plot."""

    fig, ax, params = _prepare(format, kwargs, ax)
    contour = ax.contourf(X, Y, Z, levels=levels, cmap=params["contour_cmap"])
    fig.colorbar(contour, ax=ax)
    ax.set(xlabel="X-axis", ylabel="Y-axis", title=title)
    return _finish(fig, ax, output_path, show, params["dpi"])


def pie(
    data: Any,
    labels: Any,
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a pie chart from normal non-negative counts."""

    _same_length(data, labels, "data", "labels")
    if any(float(value) < 0 for value in data):
        raise ValueError("Pie-chart counts must be non-negative.")
    if sum(float(value) for value in data) <= 0:
        raise ValueError("Pie-chart counts must have a positive total.")
    fig, ax, params = _prepare(format, kwargs, ax)
    ax.pie(
        data,
        labels=labels,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": params["font_size"]},
    )
    ax.set_title(title or "")
    return _finish(fig, ax, output_path, show, params["dpi"])


def hexbin(
    x: Any,
    y: Any,
    gridsize: int = 20,
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a hexagonal-binned density plot."""

    _same_length(x, y, "x", "y")
    fig, ax, params = _prepare(format, kwargs, ax)
    collection = ax.hexbin(x, y, gridsize=gridsize, cmap=params["hexbin_cmap"])
    fig.colorbar(collection, ax=ax, label="Counts")
    ax.set(xlabel="X-axis", ylabel="Y-axis", title=title)
    return _finish(fig, ax, output_path, show, params["dpi"])


def pairplot(
    data: Any,
    variables: Any,
    hue: str | None = None,
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    show: bool = True,
    **kwargs: Any,
) -> Any:
    """Create and return a Seaborn PairGrid."""

    warnings.warn(
        "researchplot.plots helpers are deprecated; use researchplot.use() with native plotting calls.",
        FutureWarning,
        stacklevel=2,
    )
    sns = _require("seaborn", "pairplot")
    pd = _require("pandas", "pairplot")
    params = PlotStyle(format).update(**kwargs).params
    frame = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data, columns=variables)
    selected = list(variables)
    if hue is not None and hue not in selected:
        selected.append(hue)
    with sns.plotting_context(
        "paper", font_scale=params["font_scale"], rc={"lines.linewidth": params["linewidth"]}
    ):
        grid = sns.pairplot(frame[selected], vars=list(variables), hue=hue)
    if title:
        grid.fig.suptitle(title, fontsize=params["title_font_size"], y=1.02)
    if output_path is not None:
        grid.fig.savefig(output_path, dpi=params["dpi"], bbox_inches="tight")
    if show:
        plt.show()
    return grid


def learning_curves(
    train_sizes: Any,
    train_scores: Any,
    test_scores: Any,
    format: str = "ieee",
    title: str | None = None,
    output_path: str | Path | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Plot training and test scores by training-set size."""

    _same_length(train_sizes, train_scores, "train_sizes", "train_scores")
    _same_length(train_sizes, test_scores, "train_sizes", "test_scores")
    fig, ax, params = _prepare(format, kwargs, ax)
    ax.plot(train_sizes, train_scores, label="Training Score")
    ax.plot(train_sizes, test_scores, label="Test Score")
    ax.set(xlabel="Training size", ylabel="Score", title=title)
    ax.legend(fontsize=params["legend_font_size"])
    return _finish(fig, ax, output_path, show, params["dpi"])


def time_series(
    time: Any,
    data: Any,
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a time-series line plot."""

    _same_length(time, data, "time", "data")
    fig, ax, params = _prepare(format, kwargs, ax)
    ax.plot(time, data, linewidth=params["linewidth"])
    ax.set(xlabel="Time", ylabel="Data", title=title)
    ax.tick_params(axis="x", labelrotation=45)
    return _finish(fig, ax, output_path, show, params["dpi"])


def radar_chart(
    categories: Any,
    data: Any,
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a closed polar radar chart."""

    _same_length(categories, data, "categories", "data")
    if len(categories) < 3:
        raise ValueError("radar_chart requires at least three categories.")
    np = _require("numpy", "radar_chart")
    fig, ax, params = _prepare(format, kwargs, ax, projection="polar")
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
    closed_angles = np.concatenate((angles, [angles[0]]))
    values = np.asarray(data)
    closed_values = np.concatenate((values, [values[0]]))
    ax.plot(closed_angles, closed_values, linewidth=params["linewidth"])
    ax.fill(closed_angles, closed_values, alpha=0.25)
    ax.set_xticks(angles, categories)
    ax.set_title(title or "")
    return _finish(fig, ax, output_path, show, params["dpi"])


def dendrogram(
    linkage_matrix: Any,
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a SciPy hierarchical-clustering dendrogram."""

    hierarchy = _require("scipy.cluster.hierarchy", "dendrogram")
    fig, ax, params = _prepare(format, kwargs, ax)
    hierarchy.dendrogram(linkage_matrix, ax=ax)
    ax.set(xlabel="Samples", ylabel="Distance", title=title)
    return _finish(fig, ax, output_path, show, params["dpi"])


def quiver(
    X: Any,
    Y: Any,
    U: Any,
    V: Any,
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a two-dimensional vector-field plot."""

    fig, ax, params = _prepare(format, kwargs, ax)
    ax.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=1)
    ax.set(xlabel="X-axis", ylabel="Y-axis", title=title)
    return _finish(fig, ax, output_path, show, params["dpi"])


def surface_3d(
    X: Any,
    Y: Any,
    Z: Any,
    xlabel: str = "",
    ylabel: str = "",
    zlabel: str = "",
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a three-dimensional surface plot."""

    fig, ax, params = _prepare(format, kwargs, ax, projection="3d")
    surface = cast(Any, ax).plot_surface(
        X, Y, Z, cmap=params["3d_cmap"], linewidth=params["linewidth"]
    )
    fig.colorbar(surface, ax=ax)
    ax.set(xlabel=xlabel, ylabel=ylabel, zlabel=zlabel, title=title)
    return _finish(fig, ax, output_path, show, params["dpi"], tight=True)


def sankey(
    flows: Any,
    labels: Any,
    path_lengths: Any = 0.25,
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a Matplotlib Sankey flow diagram."""

    _same_length(flows, labels, "flows", "labels")
    fig, ax, params = _prepare(format, kwargs, ax)
    Sankey(flows=flows, labels=labels, pathlengths=path_lengths, ax=ax).finish()
    ax.set_title(title or "")
    return _finish(fig, ax, output_path, show, params["dpi"])


def error_band(
    x: Any,
    y_mean: Any,
    y_std: Any,
    format: str = "ieee",
    output_path: str | Path | None = None,
    title: str | None = None,
    *,
    ax: Axes | None = None,
    show: bool = True,
    **kwargs: Any,
) -> tuple[Figure, Axes]:
    """Create a mean line with a symmetric standard-deviation band."""

    _same_length(x, y_mean, "x", "y_mean")
    _same_length(x, y_std, "x", "y_std")
    np = _require("numpy", "error_band")
    mean = np.asarray(y_mean)
    std = np.asarray(y_std)
    fig, ax, params = _prepare(format, kwargs, ax)
    ax.plot(x, mean, linewidth=params["linewidth"])
    ax.fill_between(x, mean - std, mean + std, alpha=0.3)
    ax.set_title(title or "")
    return _finish(fig, ax, output_path, show, params["dpi"])


__all__ = [
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
]
