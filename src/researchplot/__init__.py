"""ResearchPlot: source-backed venue compliance for Matplotlib figures."""

from __future__ import annotations

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .audit import audit_file
from .export import export_figure
from .models import (
    ArtworkType,
    CheckResult,
    CheckStatus,
    ComplianceError,
    RuleLevel,
    SourceRef,
    ValidationReport,
    VenueKind,
    VenueProfile,
    VenueResolutionWarning,
    VenueRule,
)
from .registry import list_venues, resolve_venue, search_venues
from .style import StyleContext, use
from .validation import validate_figure

try:
    __version__ = version("researchplot")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"

_LEGACY_NAMES = {
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


def __getattr__(name: str) -> Any:
    """Load deprecated plotting helpers only when they are requested."""

    if name in _LEGACY_NAMES:
        return getattr(import_module(".plots", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | _LEGACY_NAMES)


__all__ = [
    "ArtworkType",
    "CheckResult",
    "CheckStatus",
    "ComplianceError",
    "RuleLevel",
    "SourceRef",
    "StyleContext",
    "ValidationReport",
    "VenueKind",
    "VenueProfile",
    "VenueResolutionWarning",
    "VenueRule",
    "audit_file",
    "export_figure",
    "list_venues",
    "resolve_venue",
    "search_venues",
    "use",
    "validate_figure",
    *_LEGACY_NAMES,
]
