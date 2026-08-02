"""Measurements for live Matplotlib figures.

The inspector does not decide compliance.  It records facts and leaves source-backed
thresholds and severity to the rule engine.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

from matplotlib import colors as mcolors
from matplotlib import font_manager
from matplotlib.collections import Collection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text

from .observations import Observation, ObservationSet

_RAINBOW_MAPS = {"gist_rainbow", "hsv", "jet", "nipy_spectral", "rainbow"}
_GENERIC_FONT_FAMILIES = {"cursive", "fantasy", "monospace", "sans-serif", "serif"}


def _visible_text(fig: Figure) -> tuple[Text, ...]:
    return tuple(
        text
        for text in fig.findobj(match=Text)
        if text.get_visible() and bool(text.get_text().strip())
    )


def _resolved_family(text: Text) -> str:
    try:
        path = font_manager.findfont(text.get_fontproperties(), fallback_to_default=True)
        return str(font_manager.FontProperties(fname=path).get_name())
    except (OSError, ValueError):
        families = text.get_fontfamily()
        return str(families[0]) if families else "unknown"


def _effective_family(text: Text) -> str:
    declared = tuple(str(family) for family in text.get_fontfamily())
    generic = next(
        (family.casefold() for family in declared if family.casefold() in _GENERIC_FONT_FAMILIES),
        None,
    )
    return generic if generic is not None else _resolved_family(text)


def _line_widths(fig: Figure) -> tuple[float, ...]:
    widths: list[float] = []
    for axes in fig.axes:
        widths.extend(float(line.get_linewidth()) for line in axes.lines if line.get_visible())
        for collection in axes.collections:
            getter = getattr(collection, "get_linewidths", None)
            if not collection.get_visible() or not callable(getter):
                continue
            widths.extend(float(width) for width in getter() if float(width) > 0)
        widths.extend(
            float(spine.get_linewidth()) for spine in axes.spines.values() if spine.get_visible()
        )
        for patch in axes.patches:
            width = patch.get_linewidth()
            if patch.get_visible() and width is not None and float(width) > 0:
                widths.append(float(width))
    return tuple(widths)


def _marker_sizes(fig: Figure) -> tuple[float, ...]:
    sizes: list[float] = []
    for axes in fig.axes:
        for line in axes.lines:
            marker = line.get_marker()
            if line.get_visible() and marker not in (None, "", "None", "none", " "):
                sizes.append(float(line.get_markersize()))
        for collection in axes.collections:
            get_sizes = getattr(collection, "get_sizes", None)
            if callable(get_sizes):
                sizes.extend(math.sqrt(float(area)) for area in get_sizes() if float(area) > 0)
    return tuple(sizes)


def _has_non_color_distinctions(lines: Iterable[Line2D]) -> bool:
    visible = [line for line in lines if line.get_visible()]
    if len(visible) < 2:
        return True
    signatures = {(line.get_linestyle(), str(line.get_marker())) for line in visible}
    return len(signatures) > 1


def _rgba(value: object, alpha: float | None = None) -> tuple[float, float, float, float]:
    rgba = mcolors.to_rgba(value)
    effective_alpha = rgba[3] if alpha is None else rgba[3] * alpha
    return rgba[0], rgba[1], rgba[2], effective_alpha


def _composite(
    foreground: tuple[float, float, float, float],
    background: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    alpha = foreground[3] + background[3] * (1.0 - foreground[3])
    if alpha <= 0:
        return 0.0, 0.0, 0.0, 0.0
    channels = tuple(
        (
            foreground[index] * foreground[3]
            + background[index] * background[3] * (1.0 - foreground[3])
        )
        / alpha
        for index in range(3)
    )
    return channels[0], channels[1], channels[2], alpha


def _relative_luminance(rgba: tuple[float, float, float, float]) -> float:
    def linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    red, green, blue = (linear(value) for value in rgba[:3])
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def contrast_ratio(foreground: object, background: object) -> float:
    """Return the WCAG relative-luminance contrast ratio for two colors."""

    fg = _composite(_rgba(foreground), _rgba(background))
    bg = _rgba(background)
    high, low = sorted((_relative_luminance(fg), _relative_luminance(bg)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _minimum_text_contrast(fig: Figure, texts: tuple[Text, ...]) -> float | None:
    ratios: list[float] = []
    for text in texts:
        axes = text.axes
        background: object = axes.get_facecolor() if axes is not None else fig.get_facecolor()
        try:
            ratios.append(contrast_ratio(text.get_color(), background))
        except ValueError:
            continue
    return min(ratios) if ratios else None


def _colormap_names(fig: Figure) -> tuple[str, ...]:
    names: set[str] = set()
    for axes in fig.axes:
        artists: Iterable[object] = (*axes.images, *axes.collections)
        for artist in artists:
            if not isinstance(artist, Collection) and not hasattr(artist, "get_cmap"):
                continue
            getter = getattr(artist, "get_cmap", None)
            cmap = getter() if callable(getter) else None
            name = getattr(cmap, "name", None)
            if isinstance(name, str):
                names.add(name)
    return tuple(sorted(names))


def inspect_figure(fig: Figure) -> ObservationSet:
    """Inspect a live figure without mutating it."""

    texts = _visible_text(fig)
    font_sizes = tuple(float(text.get_fontsize()) for text in texts)
    declared_families = tuple(
        sorted({str(family) for text in texts for family in text.get_fontfamily()})
    )
    resolved_families = tuple(sorted({_resolved_family(text) for text in texts}))
    effective_families = tuple(sorted({_effective_family(text) for text in texts}))
    widths = _line_widths(fig)
    markers = _marker_sizes(fig)
    titles = tuple(
        title
        for title in (
            *(axes.get_title().strip() for axes in fig.axes),
            getattr(getattr(fig, "_suptitle", None), "get_text", lambda: "")().strip(),
        )
        if title
    )
    non_color = all(_has_non_color_distinctions(axes.lines) for axes in fig.axes)
    colormaps = _colormap_names(fig)
    text_contrast = _minimum_text_contrast(fig, texts)
    size_inches = fig.get_size_inches()

    return ObservationSet(
        (
            Observation("figure.width_mm", float(size_inches[0]) * 25.4),
            Observation("figure.height_mm", float(size_inches[1]) * 25.4),
            Observation("artifact.width_mm", float(size_inches[0]) * 25.4),
            Observation("artifact.height_mm", float(size_inches[1]) * 25.4),
            Observation("figure.dpi", float(fig.dpi)),
            Observation("font.families", declared_families, available=bool(texts)),
            Observation("font.families.resolved", resolved_families, available=bool(texts)),
            Observation("font.families.effective", effective_families, available=bool(texts)),
            Observation(
                "font.size.min_pt",
                min(font_sizes) if font_sizes else None,
                available=bool(font_sizes),
            ),
            Observation(
                "font.size.max_pt",
                max(font_sizes) if font_sizes else None,
                available=bool(font_sizes),
            ),
            Observation(
                "line.width.min_pt", min(widths) if widths else None, available=bool(widths)
            ),
            Observation(
                "marker.size.min_pt",
                min(markers) if markers else None,
                available=bool(markers),
            ),
            Observation("figure.has_title", bool(titles), detail=", ".join(titles) or None),
            Observation("color.non_color_distinctions", non_color),
            Observation(
                "accessibility.text_contrast.min",
                text_contrast,
                available=text_contrast is not None,
            ),
            Observation("accessibility.colormaps", colormaps),
            Observation(
                "accessibility.rainbow_colormap",
                any(name.casefold() in _RAINBOW_MAPS for name in colormaps),
            ),
        )
    )
