from __future__ import annotations

from pathlib import Path

import pytest
from matplotlib import pyplot as plt
from PIL import Image

from researchplot.visual_diagnostics import VisualDiagnostic, diagnose_visual


def _by_code(report: object) -> dict[str, VisualDiagnostic]:
    diagnostics = report.diagnostics  # type: ignore[attr-defined]
    return {item.code: item for item in diagnostics}


def test_raster_diagnostics_preserve_scope_and_explain_uncertainty(tmp_path: Path) -> None:
    path = tmp_path / "figure.png"
    Image.new("RGB", (20, 10), "white").save(path)

    report = diagnose_visual(path)

    assert {item.code for item in report.diagnostics} == {
        "luminance-contrast",
        "luminance-entropy",
        "dark-clipping",
        "light-clipping",
        "transparency",
    }
    assert all(0.0 <= item.confidence <= 1.0 for item in report.diagnostics)
    assert all(item.limitations for item in report.diagnostics)
    serialized = report.to_dict()["diagnostics"]
    assert all(item["limitations"] for item in serialized)  # type: ignore[index,union-attr]


def test_renderer_diagnostics_find_clipping_overlap_whitespace_and_small_text() -> None:
    figure = plt.figure(figsize=(4, 3))
    axes = figure.add_axes((0.45, 0.45, 0.1, 0.1))
    axes.plot([0, 1], [0, 1], label="trend")
    axes.text(0.5, 0.5, "first", fontsize=3)
    axes.text(0.5, 0.5, "second", fontsize=3)
    figure.text(-0.2, 0.5, "outside")
    axes.legend(loc="upper left", bbox_to_anchor=(5.0, 5.0))

    diagnostics = _by_code(diagnose_visual(figure))

    assert diagnostics["text-clipping"].value > 0
    assert diagnostics["legend-clipping"].value > 0
    assert diagnostics["overlapping-labels"].value > 0
    assert diagnostics["excess-whitespace"].value > 0.55
    assert diagnostics["final-size-legibility"].value == 3.0
    assert {
        diagnostics[code].severity
        for code in (
            "text-clipping",
            "legend-clipping",
            "overlapping-labels",
            "excess-whitespace",
            "final-size-legibility",
        )
    } == {"warning"}


def test_colormap_luminance_diagnostics_distinguish_sequential_and_rainbow() -> None:
    sequential, sequential_axes = plt.subplots(figsize=(4, 3))
    sequential_axes.imshow([[0.0, 1.0], [2.0, 3.0]], cmap="viridis")
    rainbow, rainbow_axes = plt.subplots(figsize=(4, 3))
    rainbow_axes.imshow([[0.0, 1.0], [2.0, 3.0]], cmap="jet")

    sequential_diagnostics = _by_code(diagnose_visual(sequential))
    rainbow_diagnostics = _by_code(diagnose_visual(rainbow))

    assert sequential_diagnostics["colormap-luminance-monotonicity"].value == 1.0
    assert sequential_diagnostics["colormap-luminance-monotonicity"].severity == "info"
    assert rainbow_diagnostics["colormap-luminance-monotonicity"].value < 0.95
    assert rainbow_diagnostics["colormap-luminance-monotonicity"].severity == "warning"
    assert "colormap-luminance-uniformity" in rainbow_diagnostics


def test_visual_diagnostic_rejects_unbounded_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        VisualDiagnostic("test", "info", "message", True, confidence=1.1)
