"""Deterministic accessibility previews and lightweight visual diagnostics."""

from __future__ import annotations

import hashlib
import math
import statistics
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageOps, UnidentifiedImageError

if TYPE_CHECKING:
    from matplotlib.figure import Figure

_MAX_PREVIEW_PIXELS = 100_000_000
_MAX_RENDERED_TEXT_OBJECTS = 500
_MIN_LEGIBLE_FONT_POINTS = 5.0
_OUTSIDE_TOLERANCE_PIXELS = 0.5
_OVERLAP_AREA_FRACTION = 0.15
_SIMULATION_MATRICES: dict[str, tuple[float, ...]] = {
    # Screening matrices intended to reveal colour-only encodings.  They are
    # deterministic approximations, not clinical models of individual vision.
    "protanopia": (
        0.56667,
        0.43333,
        0.0,
        0.55833,
        0.44167,
        0.0,
        0.0,
        0.24167,
        0.75833,
    ),
    "deuteranopia": (
        0.625,
        0.375,
        0.0,
        0.7,
        0.3,
        0.0,
        0.0,
        0.3,
        0.7,
    ),
    "tritanopia": (
        0.95,
        0.05,
        0.0,
        0.0,
        0.43333,
        0.56667,
        0.0,
        0.475,
        0.525,
    ),
}


@dataclass(frozen=True, slots=True)
class PreviewImage:
    """One in-memory PNG accessibility preview."""

    name: str
    png_bytes: bytes
    sha256: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class AccessibilityPreview:
    """All deterministic previews generated from one rendered figure."""

    source_width: int
    source_height: int
    images: tuple[PreviewImage, ...]
    disclaimer: str = (
        "Colour-vision simulations are screening approximations; they do not certify "
        "accessibility or model every viewer."
    )

    def get(self, name: str) -> PreviewImage:
        for preview in self.images:
            if preview.name == name:
                return preview
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class VisualDiagnostic:
    """One measured visual signal and its interpretation."""

    code: str
    severity: str
    message: str
    value: float | int | bool
    confidence: float = 0.5
    limitations: tuple[str, ...] = (
        "This is a deterministic heuristic and requires author review.",
    )

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be a finite number between zero and one.")
        if not self.limitations or any(not item.strip() for item in self.limitations):
            raise ValueError("limitations must contain at least one non-empty statement.")

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "value": self.value,
            "confidence": self.confidence,
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class VisualDiagnostics:
    """Visual measurements plus the accessibility preview set."""

    width: int
    height: int
    diagnostics: tuple[VisualDiagnostic, ...]
    previews: AccessibilityPreview

    @property
    def warnings(self) -> tuple[VisualDiagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.severity == "warning")

    def to_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "diagnostics": [item.to_dict() for item in self.diagnostics],
            "previews": [
                {
                    "name": item.name,
                    "sha256": item.sha256,
                    "width": item.width,
                    "height": item.height,
                }
                for item in self.previews.images
            ],
            "disclaimer": self.previews.disclaimer,
        }


def _source_image(source: Figure | Image.Image | str | Path, *, dpi: float) -> Image.Image:
    if isinstance(source, Image.Image):
        image = source.copy()
    elif isinstance(source, (str, Path)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(Path(source)) as opened:
                    if (
                        opened.width <= 0
                        or opened.height <= 0
                        or opened.width * opened.height > _MAX_PREVIEW_PIXELS
                    ):
                        raise ValueError(
                            f"Preview source dimensions {opened.width}x{opened.height} "
                            "exceed the safety budget."
                        )
                    opened.load()
                    image = ImageOps.exif_transpose(opened).copy()
        except (
            OSError,
            UnidentifiedImageError,
            ValueError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise ValueError(f"Could not load raster preview source {source}: {exc}") from exc
    else:
        try:
            from matplotlib.figure import Figure as MatplotlibFigure
        except ImportError as exc:  # pragma: no cover - Matplotlib is a core dependency.
            raise TypeError("Matplotlib is required to render a Figure preview.") from exc
        if not isinstance(source, MatplotlibFigure):
            raise TypeError("source must be a Matplotlib Figure, Pillow Image, or raster path.")
        if not math.isfinite(dpi) or dpi <= 0:
            raise ValueError("dpi must be finite and positive.")
        buffer = BytesIO()
        source.savefig(buffer, format="png", dpi=dpi, metadata={"Software": "ResearchPlot"})
        buffer.seek(0)
        with Image.open(buffer) as opened:
            opened.load()
            image = opened.copy()
    if image.width <= 0 or image.height <= 0 or image.width * image.height > _MAX_PREVIEW_PIXELS:
        raise ValueError(
            f"Preview source dimensions {image.width}x{image.height} exceed the safety budget."
        )
    return image


def _png_preview(name: str, image: Image.Image) -> PreviewImage:
    output = BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    payload = output.getvalue()
    return PreviewImage(
        name=name,
        png_bytes=payload,
        sha256=hashlib.sha256(payload).hexdigest(),
        width=image.width,
        height=image.height,
    )


def _apply_matrix(image: Image.Image, matrix: tuple[float, ...]) -> Image.Image:
    affine = (
        matrix[0],
        matrix[1],
        matrix[2],
        0.0,
        matrix[3],
        matrix[4],
        matrix[5],
        0.0,
        matrix[6],
        matrix[7],
        matrix[8],
        0.0,
    )
    return image.convert("RGB", affine)


def render_accessibility_previews(
    source: Figure | Image.Image | str | Path,
    *,
    dpi: float = 144.0,
) -> AccessibilityPreview:
    """Render original, grayscale, and three colour-vision screening previews."""

    source_image = _source_image(source, dpi=dpi)
    rgb = source_image.convert("RGB")
    previews = [
        _png_preview("original", rgb),
        _png_preview("grayscale", ImageOps.grayscale(rgb)),
    ]
    previews.extend(
        _png_preview(name, _apply_matrix(rgb, matrix))
        for name, matrix in _SIMULATION_MATRICES.items()
    )
    return AccessibilityPreview(rgb.width, rgb.height, tuple(previews))


@dataclass(frozen=True, slots=True)
class _RenderedText:
    bounds: tuple[float, float, float, float]
    font_size_points: float


def _bounds_area(bounds: tuple[float, float, float, float]) -> float:
    return max(0.0, bounds[2] - bounds[0]) * max(0.0, bounds[3] - bounds[1])


def _outside_bounds(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> bool:
    tolerance = _OUTSIDE_TOLERANCE_PIXELS
    return (
        inner[0] < outer[0] - tolerance
        or inner[1] < outer[1] - tolerance
        or inner[2] > outer[2] + tolerance
        or inner[3] > outer[3] + tolerance
    )


def _overlap_fraction(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    smaller_area = min(_bounds_area(first), _bounds_area(second))
    if smaller_area <= 0.0:
        return 0.0
    return width * height / smaller_area


def _finite_bounds(artist: object, renderer: object) -> tuple[float, float, float, float] | None:
    try:
        bbox = artist.get_window_extent(renderer=renderer)  # type: ignore[attr-defined]
        bounds = tuple(float(value) for value in bbox.extents)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    if len(bounds) != 4 or not all(math.isfinite(value) for value in bounds):
        return None
    typed_bounds = (bounds[0], bounds[1], bounds[2], bounds[3])
    return typed_bounds if _bounds_area(typed_bounds) > 0.0 else None


def _relative_luminance(red: float, green: float, blue: float) -> float:
    def linear(channel: float) -> float:
        return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4

    return 0.2126 * linear(red) + 0.7152 * linear(green) + 0.0722 * linear(blue)


def _colormap_diagnostics(figure: Figure) -> tuple[VisualDiagnostic, ...]:
    colormaps: dict[str, object] = {}
    for axes in figure.axes:
        if not axes.get_visible():
            continue
        for artist in (*axes.images, *axes.collections):
            get_array = getattr(artist, "get_array", None)
            get_cmap = getattr(artist, "get_cmap", None)
            if get_array is None or get_cmap is None or get_array() is None:
                continue
            colormap = get_cmap()
            if colormap is None:
                continue
            name = str(getattr(colormap, "name", type(colormap).__name__))
            colormaps.setdefault(name, colormap)

    if not colormaps:
        return ()

    monotonic_scores: list[float] = []
    uniformity_scores: list[float] = []
    for name in sorted(colormaps):
        colormap = colormaps[name]
        luminance = []
        for index in range(256):
            red, green, blue = colormap(index / 255.0)[:3]  # type: ignore[operator]
            luminance.append(_relative_luminance(float(red), float(green), float(blue)))
        changes = [
            second - first for first, second in zip(luminance[:-1], luminance[1:], strict=True)
        ]
        tolerance = 5e-4
        increasing = sum(change >= -tolerance for change in changes) / len(changes)
        decreasing = sum(change <= tolerance for change in changes) / len(changes)
        monotonic_scores.append(max(increasing, decreasing))

        magnitudes = [abs(change) for change in changes]
        mean_change = statistics.fmean(magnitudes)
        if mean_change <= 1e-12:
            uniformity_scores.append(0.0)
        else:
            coefficient = statistics.pstdev(magnitudes) / mean_change
            uniformity_scores.append(1.0 / (1.0 + coefficient))

    names = ", ".join(sorted(colormaps))
    monotonic_score = min(monotonic_scores)
    uniformity_score = min(uniformity_scores)
    shared_limitations = (
        "The metric samples the colormap definition, not how values are distributed in the data.",
        "Diverging, cyclic, and categorical maps can be intentionally non-monotonic.",
        "Luminance alone does not establish perceptual uniformity or accessible encoding.",
    )
    return (
        VisualDiagnostic(
            "colormap-luminance-monotonicity",
            "warning" if monotonic_score < 0.95 else "info",
            f"Lowest sampled luminance-monotonicity score across {names}: {monotonic_score:.3f}.",
            round(monotonic_score, 4),
            confidence=0.85,
            limitations=shared_limitations,
        ),
        VisualDiagnostic(
            "colormap-luminance-uniformity",
            "warning" if uniformity_score < 0.5 else "info",
            f"Lowest sampled luminance-step uniformity score across {names}: {uniformity_score:.3f}.",
            round(uniformity_score, 4),
            confidence=0.75,
            limitations=shared_limitations,
        ),
    )


def _rendered_figure_diagnostics(figure: Figure) -> tuple[VisualDiagnostic, ...]:
    try:
        from matplotlib.legend import Legend
        from matplotlib.text import Text

        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
    except (AttributeError, ImportError, RuntimeError, ValueError):
        return (
            VisualDiagnostic(
                "renderer-availability",
                "info",
                "The active Matplotlib canvas did not provide a renderer; layout checks were skipped.",
                False,
                confidence=1.0,
                limitations=(
                    "No clipping, overlap, whitespace, or rendered-text conclusion was produced.",
                ),
            ),
        )

    figure_bounds_raw = tuple(float(value) for value in figure.bbox.extents)
    figure_bounds = (
        figure_bounds_raw[0],
        figure_bounds_raw[1],
        figure_bounds_raw[2],
        figure_bounds_raw[3],
    )
    rendered_text: list[_RenderedText] = []
    for artist in figure.findobj(match=Text):
        if len(rendered_text) >= _MAX_RENDERED_TEXT_OBJECTS:
            break
        axes = getattr(artist, "axes", None)
        alpha = artist.get_alpha()
        if (
            not artist.get_visible()
            or (axes is not None and not axes.get_visible())
            or alpha == 0
            or not artist.get_text().strip()
        ):
            continue
        bounds = _finite_bounds(artist, renderer)
        if bounds is None:
            continue
        rendered_text.append(_RenderedText(bounds, float(artist.get_fontsize())))

    clipped_text = sum(_outside_bounds(item.bounds, figure_bounds) for item in rendered_text)
    text_limitations = (
        "Bounding boxes cannot determine whether an outside annotation was intentional.",
        "Backend font substitution and the final publication renderer can change glyph extents.",
    )
    diagnostics: list[VisualDiagnostic] = [
        VisualDiagnostic(
            "text-clipping",
            "warning" if clipped_text else "info",
            (
                f"{clipped_text} visible text object(s) extend beyond the figure canvas."
                if clipped_text
                else "No measured visible text bounds extend beyond the figure canvas."
            ),
            clipped_text,
            confidence=0.9,
            limitations=text_limitations,
        )
    ]

    legends: list[object] = list(figure.legends)
    legends.extend(legend for axes in figure.axes if (legend := axes.get_legend()) is not None)
    unique_legends = {id(legend): legend for legend in legends}
    clipped_legends = 0
    for legend in unique_legends.values():
        if not isinstance(legend, Legend) or not legend.get_visible():
            continue
        bounds = _finite_bounds(legend, renderer)
        if bounds is not None and _outside_bounds(bounds, figure_bounds):
            clipped_legends += 1
    diagnostics.append(
        VisualDiagnostic(
            "legend-clipping",
            "warning" if clipped_legends else "info",
            (
                f"{clipped_legends} visible legend(s) extend beyond the figure canvas."
                if clipped_legends
                else "No measured visible legend bounds extend beyond the figure canvas."
            ),
            clipped_legends,
            confidence=0.9,
            limitations=(
                "Bounding boxes do not detect clipping performed inside custom legend artists.",
                "A later save with tight bounding boxes can change the exported canvas.",
            ),
        )
    )

    overlaps = 0
    for index, first in enumerate(rendered_text):
        for second in rendered_text[index + 1 :]:
            if _overlap_fraction(first.bounds, second.bounds) >= _OVERLAP_AREA_FRACTION:
                overlaps += 1
    diagnostics.append(
        VisualDiagnostic(
            "overlapping-labels",
            "warning" if overlaps else "info",
            (
                f"Detected {overlaps} substantial visible-text bounding-box overlap(s)."
                if overlaps
                else "No substantial overlap was detected among visible text bounds."
            ),
            overlaps,
            confidence=0.75,
            limitations=(
                "Intentional annotations and mathematical typesetting can produce valid overlaps.",
                f"At most {_MAX_RENDERED_TEXT_OBJECTS} visible text objects are compared.",
                "Bounding-box overlap does not prove that painted glyphs collide.",
            ),
        )
    )

    axes_bounds = [
        bounds
        for axes in figure.axes
        if axes.get_visible() and (bounds := _finite_bounds(axes, renderer)) is not None
    ]
    if axes_bounds:
        occupied_bounds = (
            max(figure_bounds[0], min(bounds[0] for bounds in axes_bounds)),
            max(figure_bounds[1], min(bounds[1] for bounds in axes_bounds)),
            min(figure_bounds[2], max(bounds[2] for bounds in axes_bounds)),
            min(figure_bounds[3], max(bounds[3] for bounds in axes_bounds)),
        )
        figure_area = _bounds_area(figure_bounds)
        occupied_fraction = _bounds_area(occupied_bounds) / figure_area if figure_area else 0.0
        whitespace_fraction = 1.0 - min(1.0, occupied_fraction)
    else:
        whitespace_fraction = 1.0
    diagnostics.append(
        VisualDiagnostic(
            "excess-whitespace",
            "warning" if whitespace_fraction > 0.55 else "info",
            (
                "Axes occupy a small fraction of the rendered canvas; inspect excess whitespace."
                if whitespace_fraction > 0.55
                else "The axes bounding region occupies a substantial fraction of the canvas."
            ),
            round(whitespace_fraction, 4),
            confidence=0.65,
            limitations=(
                "The union bounding box overestimates occupied area for separated panels.",
                "Intentional margins, annotations, and publication assembly are not understood.",
            ),
        )
    )

    if rendered_text:
        minimum_font_size = min(item.font_size_points for item in rendered_text)
        undersized = sum(item.font_size_points < _MIN_LEGIBLE_FONT_POINTS for item in rendered_text)
        legibility_message = (
            f"{undersized} visible text object(s) use less than "
            f"{_MIN_LEGIBLE_FONT_POINTS:g} pt at final figure size."
            if undersized
            else f"All measured visible text uses at least {_MIN_LEGIBLE_FONT_POINTS:g} pt."
        )
    else:
        minimum_font_size = 0.0
        undersized = 0
        legibility_message = "No visible rendered text was available for a legibility prompt."
    diagnostics.append(
        VisualDiagnostic(
            "final-size-legibility",
            "warning" if undersized else "info",
            legibility_message,
            round(minimum_font_size, 3),
            confidence=0.8,
            limitations=(
                "Font point size is a proxy; typeface design, weight, contrast, and viewing distance matter.",
                "The threshold is generic accessibility guidance, not a venue requirement.",
            ),
        )
    )
    diagnostics.extend(_colormap_diagnostics(figure))
    return tuple(diagnostics)


def diagnose_visual(
    source: Figure | Image.Image | str | Path,
    *,
    dpi: float = 144.0,
) -> VisualDiagnostics:
    """Measure broad contrast, clipping, alpha, and entropy signals.

    These signals are review prompts rather than pass/fail venue rules.  They
    intentionally avoid OCR and semantic claims about labels or meaning.
    """

    figure_diagnostics: tuple[VisualDiagnostic, ...] = ()
    from matplotlib.figure import Figure as MatplotlibFigure

    if isinstance(source, MatplotlibFigure):
        figure_diagnostics = _rendered_figure_diagnostics(source)

    image = _source_image(source, dpi=dpi)
    rgba = image.convert("RGBA")
    luminance = ImageOps.grayscale(rgba.convert("RGB"))
    extrema = luminance.getextrema()
    minimum, maximum = int(extrema[0]), int(extrema[1])
    contrast_ratio = (maximum / 255.0 + 0.05) / (minimum / 255.0 + 0.05)
    histogram = luminance.histogram()
    total = image.width * image.height
    dark_fraction = sum(histogram[:3]) / total
    light_fraction = sum(histogram[253:]) / total
    alpha_histogram = rgba.getchannel("A").histogram()
    transparent_fraction = sum(alpha_histogram[:-1]) / total
    entropy = float(luminance.entropy())
    diagnostics: list[VisualDiagnostic] = [
        VisualDiagnostic(
            "luminance-contrast",
            "warning" if contrast_ratio < 3.0 else "info",
            (
                "Global luminance range is narrow; inspect labels and traces for low contrast."
                if contrast_ratio < 3.0
                else "Global luminance range provides a useful contrast span."
            ),
            round(contrast_ratio, 4),
            confidence=0.55,
            limitations=(
                "Global extrema do not measure contrast between individual text or graphical objects.",
                "A few outlier pixels can dominate this range.",
            ),
        ),
        VisualDiagnostic(
            "luminance-entropy",
            "warning" if entropy < 1.0 else "info",
            "Low entropy can indicate an empty or nearly uniform export."
            if entropy < 1.0
            else "The image is not nearly uniform by luminance entropy.",
            round(entropy, 4),
            confidence=0.65,
            limitations=(
                "A deliberately sparse figure can have low entropy without being incomplete.",
                "Entropy does not identify scientific meaning or missing plot elements.",
            ),
        ),
        VisualDiagnostic(
            "dark-clipping",
            "warning" if dark_fraction > 0.25 else "info",
            "Large near-black regions may hide clipped detail.",
            round(dark_fraction, 6),
            confidence=0.5,
            limitations=(
                "Dark backgrounds and dense imagery can legitimately contain large near-black regions.",
                "The metric does not locate or interpret individual clipped objects.",
            ),
        ),
        VisualDiagnostic(
            "light-clipping",
            "warning" if light_fraction > 0.98 else "info",
            "The export is almost entirely near-white; confirm that artists were rendered.",
            round(light_fraction, 6),
            confidence=0.6,
            limitations=(
                "Sparse plots and figures with large intentional margins can be mostly white.",
                "The metric does not establish whether all intended artists rendered.",
            ),
        ),
        VisualDiagnostic(
            "transparency",
            "info",
            "Fraction of pixels that are not fully opaque.",
            round(transparent_fraction, 6),
            confidence=0.98,
            limitations=(
                "Transparency can be intentional and permitted by the target venue.",
                "The metric does not distinguish background alpha from graphical-object alpha.",
            ),
        ),
    ]
    diagnostics.extend(figure_diagnostics)
    previews = render_accessibility_previews(image, dpi=dpi)
    return VisualDiagnostics(image.width, image.height, tuple(diagnostics), previews)


def write_accessibility_previews(
    preview: AccessibilityPreview,
    directory: str | Path,
    *,
    stem: str = "figure",
) -> tuple[Path, ...]:
    """Write previews exclusively so existing review evidence is never replaced."""

    if not stem or Path(stem).name != stem:
        raise ValueError("stem must be one safe filename component.")
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    try:
        for item in preview.images:
            path = output / f"{stem}-{item.name}.png"
            with path.open("xb") as stream:
                stream.write(item.png_bytes)
            paths.append(path)
    except BaseException:
        for path in paths:
            path.unlink(missing_ok=True)
        raise
    return tuple(paths)


__all__ = [
    "AccessibilityPreview",
    "PreviewImage",
    "VisualDiagnostic",
    "VisualDiagnostics",
    "diagnose_visual",
    "render_accessibility_previews",
    "write_accessibility_previews",
]
