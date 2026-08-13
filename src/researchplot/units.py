"""Small, deterministic unit system used by declarative profile rules.

ResearchPlot intentionally does not depend on a general-purpose units package.
The profile language needs only physical length, typography, raster density,
file size, ratios, and dimensionless counts.  Keeping that catalog closed
makes profile validation reproducible across installations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class UnitDimension(StrEnum):
    NONE = "none"
    LENGTH = "length"
    TYPOGRAPHY = "typography"
    PIXEL = "pixel"
    RESOLUTION = "resolution"
    FILE_SIZE = "file_size"
    RATIO = "ratio"
    COUNT = "count"
    ANGLE = "angle"
    BIT_DEPTH = "bit_depth"


@dataclass(frozen=True, slots=True)
class UnitDefinition:
    symbol: str
    dimension: UnitDimension
    to_base: float
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Quantity:
    """A finite numeric value with a catalogued unit."""

    value: float
    unit: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.value):
            raise ValueError("Quantity value must be finite.")
        normalize_unit(self.unit)

    def to(self, unit: str) -> Quantity:
        return Quantity(convert_value(self.value, self.unit, unit), normalize_unit(unit))


_DEFINITIONS = (
    UnitDefinition("mm", UnitDimension.LENGTH, 1.0, ("millimetre", "millimeter")),
    UnitDefinition("cm", UnitDimension.LENGTH, 10.0, ("centimetre", "centimeter")),
    UnitDefinition("in", UnitDimension.LENGTH, 25.4, ("inch", "inches")),
    UnitDefinition("pt", UnitDimension.LENGTH, 25.4 / 72.0, ("point", "points")),
    UnitDefinition("px", UnitDimension.PIXEL, 1.0, ("pixel", "pixels")),
    UnitDefinition("dpi", UnitDimension.RESOLUTION, 1.0, ("ppi",)),
    UnitDefinition("B", UnitDimension.FILE_SIZE, 1.0, ("byte", "bytes")),
    UnitDefinition("kB", UnitDimension.FILE_SIZE, 1000.0, ("KB", "kilobyte", "kilobytes")),
    UnitDefinition("MB", UnitDimension.FILE_SIZE, 1000.0 * 1000.0, ("megabyte", "megabytes")),
    UnitDefinition("KiB", UnitDimension.FILE_SIZE, 1024.0, ("kib",)),
    UnitDefinition("MiB", UnitDimension.FILE_SIZE, 1024.0 * 1024.0, ("mib",)),
    UnitDefinition("ratio", UnitDimension.RATIO, 1.0, ("fraction",)),
    UnitDefinition("percent", UnitDimension.RATIO, 0.01, ("%",)),
    UnitDefinition(
        "count",
        UnitDimension.COUNT,
        1.0,
        (
            "items",
            "page",
            "pages",
            "frame",
            "frames",
            "font",
            "fonts",
            "resource",
            "resources",
            "object",
            "objects",
            "file",
            "files",
            "link",
            "links",
            "element",
            "elements",
            "attribute",
            "attributes",
        ),
    ),
    UnitDefinition("deg", UnitDimension.ANGLE, 1.0, ("degree", "degrees")),
    UnitDefinition("bit/channel", UnitDimension.BIT_DEPTH, 1.0, ("bits/channel",)),
)

_BY_CANONICAL = {item.symbol: item for item in _DEFINITIONS}
_ALIASES = {
    alias.casefold(): item.symbol for item in _DEFINITIONS for alias in (item.symbol, *item.aliases)
}


def list_units() -> tuple[UnitDefinition, ...]:
    """Return the immutable built-in unit catalog."""

    return _DEFINITIONS


def normalize_unit(unit: str) -> str:
    """Return a canonical unit symbol or raise an actionable error."""

    if not isinstance(unit, str) or not unit.strip():
        raise ValueError("Unit must be a non-empty string.")
    canonical = _ALIASES.get(unit.strip().casefold())
    if canonical is None:
        choices = ", ".join(item.symbol for item in _DEFINITIONS)
        raise ValueError(f"Unknown unit {unit!r}; supported units: {choices}.")
    return canonical


def unit_dimension(unit: str | None) -> UnitDimension:
    """Return the dimension represented by *unit*."""

    if unit is None:
        return UnitDimension.NONE
    return _BY_CANONICAL[normalize_unit(unit)].dimension


def convert_value(value: float, from_unit: str, to_unit: str) -> float:
    """Convert a finite value between compatible catalog units."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("Unit conversion requires a numeric value.")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Unit conversion requires a finite value.")
    source = _BY_CANONICAL[normalize_unit(from_unit)]
    target = _BY_CANONICAL[normalize_unit(to_unit)]
    if source.dimension is not target.dimension:
        raise ValueError(
            f"Cannot convert {source.dimension.value} unit {source.symbol!r} "
            f"to {target.dimension.value} unit {target.symbol!r}."
        )
    return number * source.to_base / target.to_base


def require_dimension(unit: str | None, expected: UnitDimension, *, label: str) -> None:
    """Validate that a nullable unit matches a probe's declared dimension."""

    actual = unit_dimension(unit)
    if actual is not expected:
        expected_label = "no unit" if expected is UnitDimension.NONE else expected.value
        actual_label = "no unit" if actual is UnitDimension.NONE else actual.value
        raise ValueError(f"{label} requires {expected_label}; received {actual_label}.")
