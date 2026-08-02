"""Unverified style settings retained for the deprecated plotting helpers."""

from __future__ import annotations

from typing import Any

_BASE: dict[str, Any] = {
    "figsize": (6.0, 4.0),
    "dpi": 300,
    "font_size": 9,
    "font_scale": 1.0,
    "legend_font_size": 8,
    "xlabel_font_size": 9,
    "ylabel_font_size": 9,
    "title_font_size": 10,
    "linewidth": 1.0,
    "marker_size": 5,
    "xtick_font_size": 8,
    "heatmap_cmap": "viridis",
    "confusion_matrix_cmap": "Blues",
    "accuracy_loss_linewidth": 1.0,
    "contour_cmap": "viridis",
    "hexbin_cmap": "inferno",
    "3d_cmap": "viridis",
    "box_width": 0.6,
    "hist_bins": 10,
    "legend_loc": "best",
    "use_latex": False,
}


def _style(**changes: Any) -> dict[str, Any]:
    result = _BASE.copy()
    result.update(changes)
    return result


# These mappings preserve the old ``format=`` API. Only ieee/nature/elsevier are
# backed by modern profiles; the remaining names are explicitly unverified.
PLOT_FORMATS: dict[str, dict[str, Any]] = {
    "ieee": _style(figsize=(3.5, 2.2), font_size=8, linewidth=0.8),
    "nature": _style(
        figsize=(89 / 25.4, 55 / 25.4),
        dpi=600,
        font_size=7,
        legend_font_size=7,
        xlabel_font_size=7,
        ylabel_font_size=7,
        title_font_size=7,
        linewidth=0.5,
        marker_size=4,
        heatmap_cmap="inferno",
        confusion_matrix_cmap="Oranges",
        box_width=0.5,
    ),
    "elsevier": _style(figsize=(90 / 25.4, 58 / 25.4), font_size=8),
    "springer": _style(figsize=(7, 5), heatmap_cmap="magma", confusion_matrix_cmap="Purples"),
    "science": _style(
        figsize=(7, 5), font_size=10, heatmap_cmap="cividis", confusion_matrix_cmap="Reds"
    ),
    "cell": _style(figsize=(6, 4.5), dpi=600, heatmap_cmap="viridis"),
    "pnas": _style(
        figsize=(7, 5), font_size=9, heatmap_cmap="Spectral", confusion_matrix_cmap="Greys"
    ),
    "default": _style(),
}
