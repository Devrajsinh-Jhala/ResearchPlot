"""Strict, source-aware Matplotlib figure export."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure

from .models import (
    ArtworkType,
    CheckResult,
    CheckStatus,
    ComplianceError,
    RuleLevel,
    VenueProfile,
)
from .registry import resolve_venue
from .validation import allowed_formats, coerce_artwork, minimum_dpi, validate_figure

_FORMAT_ALIASES = {"jpg": "jpeg", "tif": "tiff"}
_RASTER_FORMATS = {"jpeg", "png", "tiff"}


def _normalize_format(value: str) -> str:
    normalized = value.casefold().lstrip(".")
    return _FORMAT_ALIASES.get(normalized, normalized)


def _output_paths(
    target: str | Path,
    formats: Sequence[str] | None,
    permitted: tuple[str, ...],
) -> tuple[Path, ...]:
    path = Path(target)
    requested: tuple[str, ...]
    if path.suffix:
        if formats:
            raise ValueError("Do not provide formats when the target already has a suffix.")
        requested = (_normalize_format(path.suffix),)
        base = path.with_suffix("")
    else:
        requested = tuple(_normalize_format(item) for item in formats) if formats else permitted
        base = path
    if not requested:
        raise ValueError("This profile does not specify formats for the selected artwork type.")
    if path.suffix:
        return (path,)
    return tuple(base.with_suffix(f".{item}") for item in requested)


def export_figure(
    fig: Figure,
    target: str | Path,
    *,
    venue: str | VenueProfile,
    width: str | None = None,
    artwork: ArtworkType | str = ArtworkType.VECTOR,
    formats: Sequence[str] | None = None,
    strict: bool = True,
    dpi: int | None = None,
    **savefig_kwargs: Any,
) -> tuple[Path, ...]:
    """Validate and export a figure, blocking required failures by default."""

    profile = resolve_venue(venue)
    selected_width = width or profile.default_width
    artwork_type = coerce_artwork(artwork)
    permitted = allowed_formats(profile, artwork_type)
    outputs = _output_paths(target, formats, permitted)
    required_dpi = minimum_dpi(profile, artwork_type)
    output_dpi = int(dpi if dpi is not None else required_dpi or fig.dpi)

    supported = set(fig.canvas.get_supported_filetypes())
    unsupported_by_matplotlib = [
        _normalize_format(output.suffix)
        for output in outputs
        if _normalize_format(output.suffix) not in supported
    ]
    if unsupported_by_matplotlib:
        raise ValueError(
            f"Matplotlib cannot export: {', '.join(unsupported_by_matplotlib)}. "
            f"Supported formats: {', '.join(sorted(supported))}."
        )

    original_dpi = fig.dpi
    try:
        if required_dpi is not None:
            fig.set_dpi(output_dpi)
        report = validate_figure(
            fig,
            venue=profile,
            width=selected_width,
            artwork=artwork_type,
        )
        format_rule = profile.get_rule(f"export.formats.{artwork_type.value}")
        if format_rule is not None:
            permitted_set = {_normalize_format(item) for item in permitted}
            format_checks: list[CheckResult] = []
            for output in outputs:
                output_format = _normalize_format(output.suffix)
                passed = output_format in permitted_set
                if passed:
                    status = CheckStatus.PASS
                elif format_rule.level is RuleLevel.REQUIRED:
                    status = CheckStatus.FAIL
                elif format_rule.level is RuleLevel.RECOMMENDED:
                    status = CheckStatus.WARN
                else:
                    status = CheckStatus.INFO
                format_checks.append(
                    CheckResult(
                        rule_id=format_rule.id,
                        status=status,
                        level=format_rule.level,
                        message=(
                            f"{output_format.upper()} is an allowed {artwork_type.value} format."
                            if passed
                            else f"{output_format.upper()} is not listed for {artwork_type.value} artwork."
                        ),
                        observed=output_format,
                        expected=list(permitted),
                        source_urls=profile.source_urls_for(format_rule),
                        suggestion=f"Export as one of: {', '.join(permitted)}.",
                    )
                )
            retained = tuple(check for check in report.checks if check.rule_id != format_rule.id)
            report = replace(report, checks=retained + tuple(format_checks))
        if strict and not report.passed:
            raise ComplianceError(report)
        for check in (*report.failures, *report.warnings):
            warnings.warn(check.message, UserWarning, stacklevel=2)
        written: list[Path] = []
        for output in outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            output_format = _normalize_format(output.suffix)
            options = dict(savefig_kwargs)
            options.setdefault("format", output_format)
            options.setdefault("bbox_inches", None)
            if output_format in _RASTER_FORMATS:
                options.setdefault("dpi", output_dpi)
            fig.savefig(output, **options)
            written.append(output)
        return tuple(written)
    finally:
        fig.set_dpi(original_dpi)
