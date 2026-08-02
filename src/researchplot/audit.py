"""Offline compliance audits for exported figure files."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from PIL import Image
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from .models import (
    ArtworkType,
    CheckResult,
    CheckStatus,
    RuleLevel,
    ValidationReport,
    VenueProfile,
    VenueRule,
)
from .registry import resolve_venue
from .validation import WIDTH_TOLERANCE_MM, allowed_formats, coerce_artwork, minimum_dpi

_FORMAT_ALIASES = {"jpg": "jpeg", "tif": "tiff"}
_RASTER_FORMATS = {"png", "jpeg", "tiff"}
_LENGTH = re.compile(r"^\s*([0-9.+-]+)\s*(mm|cm|in|pt|px)?\s*$", re.IGNORECASE)


def _status(rule: VenueRule, passed: bool) -> CheckStatus:
    if passed:
        return CheckStatus.PASS
    if rule.level is RuleLevel.REQUIRED:
        return CheckStatus.FAIL
    if rule.level is RuleLevel.RECOMMENDED:
        return CheckStatus.WARN
    return CheckStatus.INFO


def _check(
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


def _skip(profile: VenueProfile, rule: VenueRule, message: str) -> CheckResult:
    return CheckResult(
        rule_id=rule.id,
        status=CheckStatus.SKIP,
        level=rule.level,
        message=message,
        source_urls=profile.source_urls_for(rule),
    )


def _normalize_format(suffix: str) -> str:
    value = suffix.casefold().lstrip(".")
    return _FORMAT_ALIASES.get(value, value)


def _length_mm(value: str | None) -> float | None:
    if value is None:
        return None
    match = _LENGTH.match(value)
    if match is None:
        return None
    number = float(match.group(1))
    unit = (match.group(2) or "px").casefold()
    factors = {"mm": 1.0, "cm": 10.0, "in": 25.4, "pt": 25.4 / 72.0, "px": 25.4 / 96.0}
    return number * factors[unit]


def _pdf_measurements(path: Path) -> tuple[float, float, dict[str, Any]]:
    reader = PdfReader(path)
    if not reader.pages:
        raise ValueError("PDF contains no pages.")
    page = reader.pages[0]
    width_mm = float(page.mediabox.width) * 25.4 / 72.0
    height_mm = float(page.mediabox.height) * 25.4 / 72.0
    font_info = {"fonts": 0, "type3": 0, "unembedded": 0}
    resources = page.get("/Resources")
    if resources is not None:
        resources = resources.get_object()
        fonts = resources.get("/Font")
        if fonts is not None:
            for font_ref in fonts.get_object().values():
                font = font_ref.get_object()
                font_info["fonts"] += 1
                if str(font.get("/Subtype")) == "/Type3":
                    font_info["type3"] += 1
                descriptor = font.get("/FontDescriptor")
                if descriptor is None:
                    descendants = font.get("/DescendantFonts")
                    if descendants:
                        descriptor = descendants[0].get_object().get("/FontDescriptor")
                if descriptor is None:
                    font_info["unembedded"] += 1
                else:
                    descriptor = descriptor.get_object()
                    if not any(
                        key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")
                    ):
                        font_info["unembedded"] += 1
    return width_mm, height_mm, font_info


def _svg_measurements(path: Path) -> tuple[float | None, float | None, bool]:
    root = ET.parse(path).getroot()
    width = _length_mm(root.get("width"))
    height = _length_mm(root.get("height"))
    if (width is None or height is None) and root.get("viewBox"):
        values = [float(item) for item in root.get("viewBox", "").replace(",", " ").split()]
        if len(values) == 4:
            width = width if width is not None else values[2] * 25.4 / 96.0
            height = height if height is not None else values[3] * 25.4 / 96.0
    has_text = any(element.tag.rsplit("}", 1)[-1] == "text" for element in root.iter())
    return width, height, has_text


def _raster_measurements(path: Path) -> tuple[float | None, float | None, dict[str, Any]]:
    with Image.open(path) as image:
        dpi_value = image.info.get("dpi")
        if isinstance(dpi_value, tuple) and dpi_value:
            dpi_x = float(dpi_value[0])
            dpi_y = float(dpi_value[1] if len(dpi_value) > 1 else dpi_value[0])
        else:
            dpi_x = dpi_y = 0.0
        width_mm = image.width / dpi_x * 25.4 if dpi_x > 0 else None
        height_mm = image.height / dpi_y * 25.4 if dpi_y > 0 else None
        metadata = {
            "pixels": [image.width, image.height],
            "dpi": [dpi_x, dpi_y] if dpi_x > 0 and dpi_y > 0 else None,
            "mode": image.mode,
        }
    return width_mm, height_mm, metadata


def _eps_measurements(path: Path) -> tuple[float | None, float | None]:
    with path.open("r", encoding="latin-1", errors="replace") as stream:
        for _ in range(100):
            line = stream.readline()
            if not line:
                break
            if line.startswith("%%BoundingBox:") and "(atend)" not in line:
                values = [float(item) for item in line.split(":", 1)[1].split()]
                if len(values) == 4:
                    return (values[2] - values[0]) * 25.4 / 72.0, (
                        values[3] - values[1]
                    ) * 25.4 / 72.0
    return None, None


def audit_file(
    path: str | Path,
    *,
    venue: str | VenueProfile,
    width: str | None = None,
    artwork: ArtworkType | str = ArtworkType.VECTOR,
) -> ValidationReport:
    """Audit PDF, SVG, raster, or EPS output without network access."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Figure file not found: {file_path}")
    profile = resolve_venue(venue)
    selected_width = width or profile.default_width
    expected_width = profile.width_mm(selected_width)
    artwork_type = coerce_artwork(artwork)
    file_format = _normalize_format(file_path.suffix)
    supported = {"pdf", "svg", "png", "jpeg", "tiff", "eps"}
    if file_format not in supported:
        raise ValueError(f"Unsupported figure format {file_path.suffix!r}.")
    checks: list[CheckResult] = []

    format_rule = profile.get_rule(f"export.formats.{artwork_type.value}")
    permitted = allowed_formats(profile, artwork_type)
    if format_rule is not None:
        format_ok = file_format in {_normalize_format(item) for item in permitted}
        checks.append(
            _check(
                profile,
                format_rule,
                format_ok,
                f"{file_format.upper()} is allowed for {artwork_type.value}."
                if format_ok
                else f"{file_format.upper()} is not allowed for {artwork_type.value}.",
                observed=file_format,
                expected=list(permitted),
                suggestion=f"Export as one of: {', '.join(permitted)}.",
            )
        )

    width_mm: float | None = None
    height_mm: float | None = None
    details: dict[str, Any] = {}
    try:
        if file_format == "pdf":
            width_mm, height_mm, details = _pdf_measurements(file_path)
        elif file_format == "svg":
            width_mm, height_mm, has_text = _svg_measurements(file_path)
            details = {"has_text_elements": has_text}
        elif file_format in _RASTER_FORMATS:
            width_mm, height_mm, details = _raster_measurements(file_path)
        elif file_format == "eps":
            width_mm, height_mm = _eps_measurements(file_path)
    except (ET.ParseError, OSError, PyPdfError) as exc:
        raise ValueError(f"Could not read figure metadata from {file_path}: {exc}") from exc

    width_rule = profile.get_rule(f"figure.width.{selected_width}")
    if width_rule is None:
        raise ValueError(f"Profile {profile.id!r} does not define width {selected_width!r}.")
    if width_mm is None:
        checks.append(
            _skip(
                profile,
                width_rule,
                "Physical width could not be established from the file metadata.",
            )
        )
    else:
        width_ok = abs(width_mm - expected_width) <= WIDTH_TOLERANCE_MM
        checks.append(
            _check(
                profile,
                width_rule,
                width_ok,
                f"File width {width_mm:.2f} mm matches {selected_width}."
                if width_ok
                else f"File width {width_mm:.2f} mm does not match {selected_width}.",
                observed=round(width_mm, 3),
                expected=expected_width,
                suggestion=f"Export at exactly {expected_width:.3f} mm wide.",
            )
        )

    height_rule = profile.get_rule("figure.max_height")
    if height_rule is not None:
        if height_mm is None:
            checks.append(
                _skip(
                    profile,
                    height_rule,
                    "Physical height could not be established from the file metadata.",
                )
            )
        else:
            limit = float(height_rule.value) if isinstance(height_rule.value, (int, float)) else 0.0
            checks.append(
                _check(
                    profile,
                    height_rule,
                    height_mm <= limit + WIDTH_TOLERANCE_MM,
                    f"File height is {height_mm:.2f} mm.",
                    observed=round(height_mm, 3),
                    expected=f"<= {limit:g} mm",
                )
            )

    dpi_rule = profile.get_rule(f"export.min_dpi.{artwork_type.value}")
    if dpi_rule is not None:
        raster_dpi = details.get("dpi")
        if file_format not in _RASTER_FORMATS or not raster_dpi:
            checks.append(_skip(profile, dpi_rule, "Raster DPI is unavailable for this file."))
        else:
            observed_dpi = min(float(item) for item in raster_dpi)
            required_dpi = minimum_dpi(profile, artwork_type) or 0
            checks.append(
                _check(
                    profile,
                    dpi_rule,
                    observed_dpi + 0.5 >= required_dpi,
                    f"Raster resolution is {observed_dpi:.1f} DPI.",
                    observed=observed_dpi,
                    expected=f">= {required_dpi} DPI",
                )
            )

    type3_rule = profile.get_rule("font.pdf.type3.prohibited")
    if type3_rule is not None:
        if file_format == "pdf":
            type3 = int(details.get("type3", 0))
            unembedded = int(details.get("unembedded", 0))
            ok = type3 == 0 and unembedded == 0
            checks.append(
                _check(
                    profile,
                    type3_rule,
                    ok,
                    "PDF fonts are embedded and no Type 3 fonts were found."
                    if ok
                    else "PDF contains Type 3 or unembedded fonts.",
                    observed=details,
                    expected="Embedded fonts; no Type 3",
                    suggestion="Export with pdf.fonttype=42 and embed every font.",
                )
            )
        else:
            checks.append(
                _skip(profile, type3_rule, "PDF font checks do not apply to this file format.")
            )

    if file_format == "svg":
        checks.append(
            CheckResult(
                rule_id="file.svg.text",
                status=CheckStatus.INFO,
                level=RuleLevel.INFERRED,
                message=(
                    "SVG contains editable text elements."
                    if details.get("has_text_elements")
                    else "SVG contains no editable text elements; text may be converted to paths."
                ),
                observed=bool(details.get("has_text_elements")),
            )
        )
        font_rule = profile.get_rule("font.family")
        if font_rule is not None:
            checks.append(
                _skip(
                    profile,
                    font_rule,
                    "SVG text presence was inspected, but computed font families cannot be established reliably.",
                )
            )

    if file_format in _RASTER_FORMATS:
        color_mode = details.get("mode")
        checks.append(
            CheckResult(
                rule_id="file.color_mode",
                status=CheckStatus.INFO,
                level=RuleLevel.INFERRED,
                message=(
                    f"Raster color mode is {color_mode}; this profile specifies no "
                    "source-backed color-mode restriction."
                ),
                observed=color_mode,
                expected=None,
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
