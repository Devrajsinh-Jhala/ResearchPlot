"""Validation of live Matplotlib figures against venue profiles."""

from __future__ import annotations

from collections.abc import Iterable

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.text import Text

from .models import (
    ArtworkType,
    CheckResult,
    CheckStatus,
    RuleLevel,
    ValidationReport,
    VenueProfile,
    VenueRule,
)
from .registry import normalize_venue_name, resolve_venue

WIDTH_TOLERANCE_MM = 0.5


def coerce_artwork(value: ArtworkType | str) -> ArtworkType:
    """Convert a user artwork name to :class:`ArtworkType`."""

    if isinstance(value, ArtworkType):
        return value
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return ArtworkType(normalized)
    except ValueError as exc:
        choices = ", ".join(item.value for item in ArtworkType)
        raise ValueError(f"Unknown artwork type {value!r}. Choose from: {choices}.") from exc


def _status(rule: VenueRule, passed: bool) -> CheckStatus:
    if passed:
        return CheckStatus.PASS
    if rule.level is RuleLevel.REQUIRED:
        return CheckStatus.FAIL
    if rule.level is RuleLevel.RECOMMENDED:
        return CheckStatus.WARN
    return CheckStatus.INFO


def _result(
    profile: VenueProfile,
    rule: VenueRule,
    passed: bool,
    message: str,
    *,
    observed: object = None,
    expected: object = None,
    suggestion: str | None = None,
) -> CheckResult:
    return CheckResult(
        rule_id=rule.id,
        status=_status(rule, passed),
        level=rule.level,
        message=message,
        observed=observed,
        expected=expected,
        source_urls=profile.source_urls_for(rule),
        suggestion=suggestion,
    )


def _skipped(profile: VenueProfile, rule: VenueRule, message: str) -> CheckResult:
    return CheckResult(
        rule_id=rule.id,
        status=CheckStatus.SKIP,
        level=rule.level,
        message=message,
        source_urls=profile.source_urls_for(rule),
    )


def _number(rule: VenueRule) -> float:
    if not isinstance(rule.value, (int, float)):
        raise TypeError(f"Rule {rule.id!r} must contain a numeric value.")
    return float(rule.value)


def allowed_formats(profile: VenueProfile, artwork: ArtworkType | str) -> tuple[str, ...]:
    """Return normalized output formats allowed by a profile and artwork type."""

    artwork_type = coerce_artwork(artwork)
    rule = profile.get_rule(f"export.formats.{artwork_type.value}")
    if rule is None or not isinstance(rule.value, tuple):
        return ()
    return tuple(str(item).lower() for item in rule.value)


def minimum_dpi(profile: VenueProfile, artwork: ArtworkType | str) -> int | None:
    """Return the source-backed minimum raster DPI, when specified."""

    artwork_type = coerce_artwork(artwork)
    rule = profile.get_rule(f"export.min_dpi.{artwork_type.value}")
    if rule is None:
        return None
    return int(_number(rule))


def _visible_text(fig: Figure) -> tuple[Text, ...]:
    return tuple(
        text
        for text in fig.findobj(match=Text)
        if text.get_visible() and bool(text.get_text().strip())
    )


def _line_widths(fig: Figure) -> tuple[float, ...]:
    widths: list[float] = []
    for axes in fig.axes:
        widths.extend(float(line.get_linewidth()) for line in axes.lines if line.get_visible())
        widths.extend(
            float(spine.get_linewidth()) for spine in axes.spines.values() if spine.get_visible()
        )
        for patch in axes.patches:
            width = patch.get_linewidth()
            if patch.get_visible() and width is not None and float(width) > 0:
                widths.append(float(width))
    return tuple(widths)


def _marker_sizes(fig: Figure) -> tuple[float, ...]:
    marker_sizes: list[float] = []
    for axes in fig.axes:
        for line in axes.lines:
            marker = line.get_marker()
            if line.get_visible() and marker not in (None, "", "None", "none", " "):
                marker_sizes.append(float(line.get_markersize()))
        for collection in axes.collections:
            get_sizes = getattr(collection, "get_sizes", None)
            if callable(get_sizes):
                marker_sizes.extend(float(area) ** 0.5 for area in get_sizes() if float(area) > 0)
    return tuple(marker_sizes)


def _titles(fig: Figure) -> tuple[str, ...]:
    titles = [axes.get_title().strip() for axes in fig.axes if axes.get_title().strip()]
    suptitle = getattr(fig, "_suptitle", None)
    if isinstance(suptitle, Text) and suptitle.get_text().strip():
        titles.append(suptitle.get_text().strip())
    return tuple(titles)


def _uses_non_color_distinctions(lines: Iterable[Line2D]) -> bool:
    visible = [line for line in lines if line.get_visible()]
    if len(visible) < 2:
        return True
    styles = {(line.get_linestyle(), str(line.get_marker())) for line in visible}
    return len(styles) > 1


def validate_figure(
    fig: Figure,
    *,
    venue: str | VenueProfile,
    width: str | None = None,
    artwork: ArtworkType | str = ArtworkType.VECTOR,
) -> ValidationReport:
    """Validate a live Matplotlib figure without mutating it."""

    profile = resolve_venue(venue)
    selected_width = width or profile.default_width
    expected_width = profile.width_mm(selected_width)
    artwork_type = coerce_artwork(artwork)
    checks: list[CheckResult] = []

    width_rule = profile.get_rule(f"figure.width.{selected_width}")
    if width_rule is None:
        raise ValueError(f"Profile {profile.id!r} does not define width {selected_width!r}.")
    observed_width = float(fig.get_size_inches()[0]) * 25.4
    width_ok = abs(observed_width - expected_width) <= WIDTH_TOLERANCE_MM
    checks.append(
        _result(
            profile,
            width_rule,
            width_ok,
            (
                f"Figure width {observed_width:.2f} mm matches {selected_width}."
                if width_ok
                else f"Figure width {observed_width:.2f} mm does not match {selected_width}."
            ),
            observed=round(observed_width, 3),
            expected=expected_width,
            suggestion=f"Create or resize the figure to {expected_width:.3f} mm wide.",
        )
    )

    height_rule = profile.get_rule("figure.max_height")
    if height_rule is not None:
        expected_height = _number(height_rule)
        observed_height = float(fig.get_size_inches()[1]) * 25.4
        height_ok = observed_height <= expected_height + WIDTH_TOLERANCE_MM
        checks.append(
            _result(
                profile,
                height_rule,
                height_ok,
                (
                    f"Figure height {observed_height:.2f} mm is within the limit."
                    if height_ok
                    else f"Figure height {observed_height:.2f} mm exceeds the venue limit."
                ),
                observed=round(observed_height, 3),
                expected=f"<= {expected_height} mm",
                suggestion=f"Reduce the figure height to at most {expected_height:.1f} mm.",
            )
        )

    texts = _visible_text(fig)
    family_rule = profile.get_rule("font.family")
    if family_rule is not None and isinstance(family_rule.value, tuple):
        allowed = {normalize_venue_name(str(family)) for family in family_rule.value}
        observed_families = {
            normalize_venue_name(str(family)) for text in texts for family in text.get_fontfamily()
        }
        if not texts:
            checks.append(_skipped(profile, family_rule, "No visible text is available to check."))
        else:
            family_ok = all(
                bool(
                    allowed
                    & {normalize_venue_name(str(family)) for family in text.get_fontfamily()}
                )
                for text in texts
            )
            checks.append(
                _result(
                    profile,
                    family_rule,
                    family_ok,
                    (
                        "Every visible text artist uses an allowed family."
                        if family_ok
                        else "A visible text artist does not use an allowed family."
                    ),
                    observed=sorted(observed_families),
                    expected=list(family_rule.value),
                    suggestion="Use the profile context or an allowed installed font family.",
                )
            )

    sizes = tuple(float(text.get_fontsize()) for text in texts)
    min_font_rule = profile.get_rule("font.size.min")
    if min_font_rule is not None:
        expected_min = _number(min_font_rule)
        observed_min = min(sizes) if sizes else None
        if observed_min is None:
            checks.append(
                _skipped(profile, min_font_rule, "No visible text is available to check.")
            )
        else:
            font_ok = observed_min >= expected_min - 0.01
            checks.append(
                _result(
                    profile,
                    min_font_rule,
                    font_ok,
                    (
                        "All visible text meets the minimum size."
                        if font_ok
                        else f"Visible text is smaller than {expected_min:g} pt."
                    ),
                    observed=observed_min,
                    expected=f">= {expected_min:g} pt",
                    suggestion=f"Increase all visible text to at least {expected_min:g} pt.",
                )
            )

    max_font_rule = profile.get_rule("font.size.max")
    if max_font_rule is not None:
        expected_max = _number(max_font_rule)
        observed_max = max(sizes) if sizes else None
        if observed_max is None:
            checks.append(
                _skipped(profile, max_font_rule, "No visible text is available to check.")
            )
        else:
            font_ok = observed_max <= expected_max + 0.01
            checks.append(
                _result(
                    profile,
                    max_font_rule,
                    font_ok,
                    (
                        "All visible text is within the maximum size."
                        if font_ok
                        else f"Visible text is larger than {expected_max:g} pt."
                    ),
                    observed=observed_max,
                    expected=f"<= {expected_max:g} pt",
                    suggestion=f"Reduce all visible text to at most {expected_max:g} pt.",
                )
            )

    line_rule = profile.get_rule("line.width.min")
    if line_rule is not None:
        expected_line = _number(line_rule)
        widths = _line_widths(fig)
        observed_line = min(widths) if widths else None
        if observed_line is None:
            checks.append(_skipped(profile, line_rule, "No visible lines are available to check."))
        else:
            line_ok = observed_line >= expected_line - 0.001
            checks.append(
                _result(
                    profile,
                    line_rule,
                    line_ok,
                    (
                        "Visible line weights meet the venue minimum."
                        if line_ok
                        else f"A visible line is thinner than {expected_line:g} pt."
                    ),
                    observed=observed_line,
                    expected=f">= {expected_line:g} pt",
                    suggestion=f"Set visible line widths to at least {expected_line:g} pt.",
                )
            )

    marker_rule = profile.get_rule("marker.size.min")
    if marker_rule is not None:
        expected_marker = _number(marker_rule)
        marker_sizes = _marker_sizes(fig)
        observed_marker = min(marker_sizes) if marker_sizes else None
        if observed_marker is None:
            checks.append(
                _skipped(profile, marker_rule, "No visible markers are available to check.")
            )
        else:
            marker_ok = observed_marker >= expected_marker - 0.01
            checks.append(
                _result(
                    profile,
                    marker_rule,
                    marker_ok,
                    "Visible markers meet the venue minimum."
                    if marker_ok
                    else "Markers are too small.",
                    observed=observed_marker,
                    expected=f">= {expected_marker:g} pt",
                )
            )

    title_rule = profile.get_rule("figure.title.prohibited")
    if title_rule is not None and title_rule.value is True:
        titles = _titles(fig)
        title_ok = not titles
        checks.append(
            _result(
                profile,
                title_rule,
                title_ok,
                "No title is embedded in the figure."
                if title_ok
                else "A title is embedded in the figure.",
                observed=list(titles),
                expected="No in-figure title",
                suggestion="Move the title to the manuscript figure legend.",
            )
        )

    grayscale_rule = profile.get_rule("color.grayscale.distinguishable")
    if grayscale_rule is not None and grayscale_rule.value is True:
        grayscale_ok = all(_uses_non_color_distinctions(axes.lines) for axes in fig.axes)
        checks.append(
            _result(
                profile,
                grayscale_rule,
                grayscale_ok,
                (
                    "Multiple series use line style or marker distinctions."
                    if grayscale_ok
                    else "Multiple series appear to rely on color alone."
                ),
                suggestion="Combine color with distinct line styles or markers.",
            )
        )

    type3_rule = profile.get_rule("font.pdf.type3.prohibited")
    if type3_rule is not None and type3_rule.value is True:
        pdf_type = int(mpl.rcParams["pdf.fonttype"])
        type_ok = pdf_type != 3
        checks.append(
            _result(
                profile,
                type3_rule,
                type_ok,
                "PDF output is configured without Type 3 fonts."
                if type_ok
                else "PDF output uses Type 3 fonts.",
                observed=pdf_type,
                expected="Not Type 3",
                suggestion="Set matplotlib rcParam 'pdf.fonttype' to 42.",
            )
        )

    dpi_rule = profile.get_rule(f"export.min_dpi.{artwork_type.value}")
    if dpi_rule is not None:
        expected_dpi = _number(dpi_rule)
        observed_dpi = float(fig.dpi)
        dpi_ok = observed_dpi >= expected_dpi
        checks.append(
            _result(
                profile,
                dpi_rule,
                dpi_ok,
                "Figure DPI meets the raster requirement."
                if dpi_ok
                else "Figure DPI is below the raster requirement.",
                observed=observed_dpi,
                expected=f">= {expected_dpi:g} DPI",
                suggestion=f"Create or export raster artwork at {expected_dpi:g} DPI or higher.",
            )
        )

    format_rule = profile.get_rule(f"export.formats.{artwork_type.value}")
    if format_rule is not None:
        checks.append(
            _skipped(
                profile,
                format_rule,
                "Output format is not known until export or file audit.",
            )
        )

    for index, caveat in enumerate(profile.caveats):
        checks.append(
            CheckResult(
                rule_id=f"profile.caveat.{index + 1}",
                status=CheckStatus.INFO,
                level=RuleLevel.INFERRED,
                message=caveat,
            )
        )

    return ValidationReport(
        profile_id=profile.id,
        width=selected_width,
        artwork=artwork_type,
        checks=tuple(checks),
    )
