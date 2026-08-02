from __future__ import annotations

import pytest
from matplotlib import pyplot as plt

from researchplot.figure_inspector import contrast_ratio, inspect_figure


def test_wcag_contrast_formula() -> None:
    assert contrast_ratio("black", "white") == pytest.approx(21.0)
    assert contrast_ratio("white", "white") == pytest.approx(1.0)


def test_live_figure_observations() -> None:
    fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
    ax.plot([0, 1], [0, 1], marker="o", linewidth=1.25, label="A")
    ax.plot([0, 1], [1, 0], marker="s", linestyle="--", label="B")
    ax.set(xlabel="x", ylabel="y")
    observations = inspect_figure(fig)

    assert observations.get("figure.width_mm").value == pytest.approx(88.9)  # type: ignore[union-attr]
    assert observations.get("figure.dpi").value == 200  # type: ignore[union-attr]
    assert observations.get("font.size.min_pt").available  # type: ignore[union-attr]
    assert observations.get("color.non_color_distinctions").value is True  # type: ignore[union-attr]
    assert observations.get("accessibility.rainbow_colormap").value is False  # type: ignore[union-attr]


def test_rainbow_colormap_is_observed() -> None:
    fig, ax = plt.subplots()
    ax.imshow([[0, 1], [2, 3]], cmap="jet")
    observations = inspect_figure(fig)
    assert observations.get("accessibility.rainbow_colormap").value is True  # type: ignore[union-attr]
