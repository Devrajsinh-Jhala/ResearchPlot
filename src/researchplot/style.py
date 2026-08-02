"""Matplotlib-native venue style contexts."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from types import TracebackType
from typing import Any, cast

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.figure import Figure

from .models import ArtworkType, ValidationReport, VenueProfile
from .registry import resolve_venue
from .validation import validate_figure


def _numeric_rule(profile: VenueProfile, rule_id: str, default: float) -> float:
    rule = profile.get_rule(rule_id)
    if rule is None or not isinstance(rule.value, (int, float)):
        return default
    return float(rule.value)


def _font_family(profile: VenueProfile) -> str:
    rule = profile.get_rule("font.family")
    if rule is None or not isinstance(rule.value, tuple):
        return "sans-serif"
    installed = {font.name.casefold(): font.name for font in font_manager.fontManager.ttflist}
    for requested in rule.value:
        name = str(requested)
        if name.casefold() in installed:
            return cast(str, installed[name.casefold()])
    for generic in ("sans-serif", "serif", "monospace"):
        if generic in rule.value:
            return generic
    return str(rule.value[-1])


def _validated_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    if overrides is None:
        return {}
    validated: dict[str, Any] = {}
    for key, value in overrides.items():
        if key not in mpl.rcParams:
            raise ValueError(f"Unknown Matplotlib rcParam override {key!r}.")
        try:
            validated[key] = mpl.rcParams.validate[key](value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid value for Matplotlib rcParam {key!r}: {value!r}.") from exc
    return validated


class StyleContext(AbstractContextManager["StyleContext"]):
    """A reversible Matplotlib style and venue-aware figure factory."""

    def __init__(
        self,
        venue: str | VenueProfile,
        *,
        width: str | None = None,
        latex: bool = False,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.profile = resolve_venue(venue)
        self.width = width or self.profile.default_width
        self.width_mm = self.profile.width_mm(self.width)
        self.latex = bool(latex)
        self.overrides = _validated_overrides(overrides)
        self._context: AbstractContextManager[None] | None = None

    @property
    def rc(self) -> dict[str, Any]:
        """The validated rcParams applied by this context."""

        font_size = _numeric_rule(self.profile, "font.size.target", 8.0)
        line_width = _numeric_rule(self.profile, "line.width.min", 0.8)
        marker_size = _numeric_rule(self.profile, "marker.size.min", 4.0)
        settings: dict[str, Any] = {
            "font.family": _font_family(self.profile),
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "lines.linewidth": line_width,
            "lines.markersize": marker_size,
            "figure.constrained_layout.use": True,
            "savefig.bbox": None,
            "savefig.pad_inches": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": self.latex,
        }
        settings.update(self.overrides)
        return settings

    def __enter__(self) -> StyleContext:
        if self._context is not None:
            raise RuntimeError("A StyleContext cannot be entered more than once at a time.")
        self._context = mpl.rc_context(rc=self.rc)
        self._context.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        if self._context is None:
            return None
        context, self._context = self._context, None
        return context.__exit__(exc_type, exc_value, traceback)

    def figure(
        self,
        *,
        height: float | None = None,
        aspect: float = 0.62,
        **kwargs: Any,
    ) -> Figure:
        """Create a figure at the exact final venue width.

        ``height`` is expressed in millimetres. When omitted, ``aspect`` is the
        height-to-width ratio and an official maximum height is respected.
        """

        height_mm = float(height) if height is not None else self.width_mm * float(aspect)
        max_height = self.profile.get_rule("figure.max_height")
        if max_height is not None and isinstance(max_height.value, (int, float)):
            height_mm = min(height_mm, float(max_height.value))
        if height_mm <= 0:
            raise ValueError("Figure height must be greater than zero.")
        kwargs.setdefault("figsize", (self.width_mm / 25.4, height_mm / 25.4))
        return Figure(**kwargs)

    def subplots(
        self,
        *args: Any,
        height: float | None = None,
        aspect: float = 0.62,
        **kwargs: Any,
    ) -> tuple[Figure, Any]:
        """Create venue-sized subplots using the active Matplotlib backend."""

        from matplotlib import pyplot as plt

        height_mm = float(height) if height is not None else self.width_mm * float(aspect)
        max_height = self.profile.get_rule("figure.max_height")
        if max_height is not None and isinstance(max_height.value, (int, float)):
            height_mm = min(height_mm, float(max_height.value))
        if height_mm <= 0:
            raise ValueError("Figure height must be greater than zero.")
        kwargs.setdefault("figsize", (self.width_mm / 25.4, height_mm / 25.4))
        return cast(tuple[Figure, Any], plt.subplots(*args, **kwargs))

    def validate(
        self,
        fig: Figure,
        *,
        artwork: ArtworkType | str = ArtworkType.VECTOR,
    ) -> ValidationReport:
        """Validate ``fig`` against this context's resolved profile and width."""

        with mpl.rc_context(rc=self.rc):
            return validate_figure(
                fig,
                venue=self.profile,
                width=self.width,
                artwork=artwork,
            )

    def export(
        self,
        fig: Figure,
        target: str | Path,
        *,
        artwork: ArtworkType | str = ArtworkType.VECTOR,
        formats: tuple[str, ...] | list[str] | None = None,
        strict: bool = True,
        dpi: int | None = None,
        **savefig_kwargs: Any,
    ) -> tuple[Path, ...]:
        """Validate and export ``fig`` using this context's profile."""

        from .export import export_figure

        with mpl.rc_context(rc=self.rc):
            return export_figure(
                fig,
                target,
                venue=self.profile,
                width=self.width,
                artwork=artwork,
                formats=formats,
                strict=strict,
                dpi=dpi,
                **savefig_kwargs,
            )


def use(
    venue: str | VenueProfile,
    *,
    width: str | None = None,
    latex: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> StyleContext:
    """Return a reversible Matplotlib venue style context."""

    return StyleContext(venue, width=width, latex=latex, overrides=overrides)
