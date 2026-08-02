"""Matplotlib-native venue style contexts."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any, cast

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.figure import Figure

from .models import VenueProfile
from .registry import resolve_profile


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
    fallback_path = font_manager.findfont(
        font_manager.FontProperties(family=["sans-serif"]), fallback_to_default=True
    )
    fallback_name = font_manager.FontProperties(fname=fallback_path).get_name()
    return str(fallback_name or "sans-serif")


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
        self.profile = resolve_profile(venue)
        self.width = width if width is not None else self.profile.default_width
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

        height_mm = self._height_mm(height=height, aspect=aspect)
        if height_mm <= 0:
            raise ValueError("Figure height must be greater than zero.")
        expected = (self.width_mm / 25.4, height_mm / 25.4)
        supplied = kwargs.pop("figsize", None)
        if supplied is not None and tuple(float(item) for item in supplied) != expected:
            raise ValueError(
                "figsize cannot override a venue target; use height= or aspect= instead."
            )
        kwargs["figsize"] = expected
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

        height_mm = self._height_mm(height=height, aspect=aspect)
        if height_mm <= 0:
            raise ValueError("Figure height must be greater than zero.")
        expected = (self.width_mm / 25.4, height_mm / 25.4)
        supplied = kwargs.pop("figsize", None)
        if supplied is not None and tuple(float(item) for item in supplied) != expected:
            raise ValueError(
                "figsize cannot override a venue target; use height= or aspect= instead."
            )
        kwargs["figsize"] = expected
        return cast(tuple[Figure, Any], plt.subplots(*args, **kwargs))

    def _height_mm(self, *, height: float | None, aspect: float) -> float:
        if aspect <= 0:
            raise ValueError("Figure aspect must be greater than zero.")
        height_mm = float(height) if height is not None else self.width_mm * float(aspect)
        max_height = self.profile.get_rule("figure.max_height")
        if max_height is not None and isinstance(max_height.value, (int, float)):
            maximum = float(max_height.value)
            if height is not None and height_mm > maximum:
                raise ValueError(
                    f"Requested height {height_mm:g} mm exceeds the {maximum:g} mm venue limit."
                )
            height_mm = min(height_mm, maximum)
        return height_mm
