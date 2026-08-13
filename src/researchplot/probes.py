"""Closed catalog of observations understood by bundled profile rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .models import ConstraintOperator, RuleConstraint, RulePhase
from .units import UnitDimension, require_dimension


class ProbeValueKind(StrEnum):
    NUMBER = "number"
    BOOLEAN = "boolean"
    STRING = "string"
    STRING_SET = "string_set"


@dataclass(frozen=True, slots=True)
class ProbeDefinition:
    id: str
    value_kind: ProbeValueKind
    dimension: UnitDimension
    phases: tuple[RulePhase, ...]
    description: str


def _probe(
    id: str,
    value_kind: ProbeValueKind,
    dimension: UnitDimension,
    phases: tuple[RulePhase, ...],
    description: str,
) -> ProbeDefinition:
    return ProbeDefinition(id, value_kind, dimension, phases, description)


_LIVE = (RulePhase.LIVE,)
_FILE = (RulePhase.FILE,)
_BUNDLE = (RulePhase.BUNDLE,)
_LIVE_FILE = (RulePhase.LIVE, RulePhase.FILE)

_CATALOG = tuple(
    sorted(
        (
            _probe(
                "artifact.width_mm",
                ProbeValueKind.NUMBER,
                UnitDimension.LENGTH,
                _LIVE_FILE,
                "Physical artifact width.",
            ),
            _probe(
                "artifact.height_mm",
                ProbeValueKind.NUMBER,
                UnitDimension.LENGTH,
                _LIVE_FILE,
                "Physical artifact height.",
            ),
            _probe(
                "artifact.dpi",
                ProbeValueKind.NUMBER,
                UnitDimension.RESOLUTION,
                _FILE,
                "Raster density.",
            ),
            _probe(
                "artifact.file_size",
                ProbeValueKind.NUMBER,
                UnitDimension.FILE_SIZE,
                _FILE,
                "Artifact byte size.",
            ),
            _probe(
                "artifact.page_count",
                ProbeValueKind.NUMBER,
                UnitDimension.COUNT,
                _FILE,
                "Number of document pages.",
            ),
            _probe(
                "artifact.format",
                ProbeValueKind.STRING,
                UnitDimension.NONE,
                _FILE,
                "Normalized output format.",
            ),
            _probe(
                "figure.has_title",
                ProbeValueKind.BOOLEAN,
                UnitDimension.NONE,
                _LIVE,
                "Whether an axes title is present.",
            ),
            _probe(
                "font.families.effective",
                ProbeValueKind.STRING_SET,
                UnitDimension.NONE,
                _LIVE,
                "Effective rendered font families.",
            ),
            _probe(
                "font.size.min_pt",
                ProbeValueKind.NUMBER,
                UnitDimension.LENGTH,
                _LIVE,
                "Smallest rendered text size.",
            ),
            _probe(
                "font.size.max_pt",
                ProbeValueKind.NUMBER,
                UnitDimension.LENGTH,
                _LIVE,
                "Largest rendered text size.",
            ),
            _probe(
                "line.width.min_pt",
                ProbeValueKind.NUMBER,
                UnitDimension.LENGTH,
                _LIVE,
                "Smallest rendered stroke width.",
            ),
            _probe(
                "marker.size.min_pt",
                ProbeValueKind.NUMBER,
                UnitDimension.LENGTH,
                _LIVE,
                "Smallest rendered marker size.",
            ),
            _probe(
                "pdf.type3_font_count",
                ProbeValueKind.NUMBER,
                UnitDimension.COUNT,
                _FILE,
                "Type 3 fonts in a PDF.",
            ),
            _probe(
                "pdf.unembedded_font_count",
                ProbeValueKind.NUMBER,
                UnitDimension.COUNT,
                _FILE,
                "Unembedded fonts in a PDF.",
            ),
            _probe(
                "pdf.unembedded_truetype_font_count",
                ProbeValueKind.NUMBER,
                UnitDimension.COUNT,
                _FILE,
                "Unembedded TrueType fonts in a PDF.",
            ),
            _probe(
                "raster.mode",
                ProbeValueKind.STRING,
                UnitDimension.NONE,
                _FILE,
                "Raster color mode.",
            ),
            _probe(
                "raster.compression",
                ProbeValueKind.STRING,
                UnitDimension.NONE,
                _FILE,
                "Raster compression scheme.",
            ),
            _probe(
                "color.non_color_distinctions",
                ProbeValueKind.BOOLEAN,
                UnitDimension.NONE,
                _LIVE,
                "Whether series use distinctions in addition to color.",
            ),
            _probe(
                "accessibility.minimum_contrast",
                ProbeValueKind.NUMBER,
                UnitDimension.RATIO,
                _LIVE,
                "Minimum meaningful graphical contrast ratio.",
            ),
            _probe(
                "accessibility.colormaps",
                ProbeValueKind.STRING_SET,
                UnitDimension.NONE,
                _LIVE,
                "Colormaps used in the figure.",
            ),
            _probe(
                "metadata.alt_text.present",
                ProbeValueKind.BOOLEAN,
                UnitDimension.NONE,
                _BUNDLE,
                "Whether alt text is supplied.",
            ),
            _probe(
                "metadata.alt_text.distinct_from_caption",
                ProbeValueKind.BOOLEAN,
                UnitDimension.NONE,
                _BUNDLE,
                "Whether alt text adds information beyond the caption.",
            ),
            _probe(
                "metadata.caption.present",
                ProbeValueKind.BOOLEAN,
                UnitDimension.NONE,
                _BUNDLE,
                "Whether a caption is supplied.",
            ),
            _probe(
                "metadata.source_data.present",
                ProbeValueKind.BOOLEAN,
                UnitDimension.NONE,
                _BUNDLE,
                "Whether source data is associated.",
            ),
            _probe(
                "metadata.long_description.present",
                ProbeValueKind.BOOLEAN,
                UnitDimension.NONE,
                _BUNDLE,
                "Whether a long description is supplied.",
            ),
            _probe(
                "metadata.key_trends.present",
                ProbeValueKind.BOOLEAN,
                UnitDimension.NONE,
                _BUNDLE,
                "Whether key trends are supplied.",
            ),
            _probe(
                "metadata.data_table.present",
                ProbeValueKind.BOOLEAN,
                UnitDimension.NONE,
                _BUNDLE,
                "Whether an accessible data table is supplied.",
            ),
            _probe(
                "metadata.attachments.present",
                ProbeValueKind.BOOLEAN,
                UnitDimension.NONE,
                _BUNDLE,
                "Whether declared attachments are present.",
            ),
            _probe(
                "metadata.panel_descriptions.complete",
                ProbeValueKind.BOOLEAN,
                UnitDimension.NONE,
                _BUNDLE,
                "Whether every panel has a description.",
            ),
            _probe(
                "metadata.panel_order.complete",
                ProbeValueKind.BOOLEAN,
                UnitDimension.NONE,
                _BUNDLE,
                "Whether every panel has an explicit order.",
            ),
            _probe(
                "metadata.figure_number.present",
                ProbeValueKind.BOOLEAN,
                UnitDimension.NONE,
                _BUNDLE,
                "Whether a stable figure number is supplied.",
            ),
            _probe(
                "metadata.filename.compliant",
                ProbeValueKind.BOOLEAN,
                UnitDimension.NONE,
                _BUNDLE,
                "Whether the artifact filename matches venue rules.",
            ),
        ),
        key=lambda item: item.id,
    )
)

_BY_ID = {item.id: item for item in _CATALOG}


def list_probes() -> tuple[ProbeDefinition, ...]:
    return _CATALOG


def get_probe(probe_id: str) -> ProbeDefinition | None:
    return _BY_ID.get(probe_id)


def validate_probe_constraint(
    probe_id: str,
    constraint: RuleConstraint,
    phases: tuple[RulePhase, ...],
    *,
    label: str,
) -> None:
    """Validate value shape, units, and phases for a known probe."""

    definition = get_probe(probe_id)
    if definition is None:
        choices = ", ".join(item.id for item in _CATALOG)
        raise ValueError(f"{label} uses unknown probe {probe_id!r}; known probes: {choices}.")
    invalid_phases = set(phases) - set(definition.phases)
    if invalid_phases:
        names = ", ".join(sorted(item.value for item in invalid_phases))
        raise ValueError(f"{label} cannot evaluate {probe_id!r} during: {names}.")

    expected_dimension = definition.dimension
    unit = constraint.unit
    if expected_dimension in {UnitDimension.COUNT, UnitDimension.NONE} and unit is None:
        pass
    else:
        require_dimension(unit, expected_dimension, label=f"{label}.constraint.unit")

    value = constraint.value
    scalar_values = value if isinstance(value, tuple) else (value,)
    if definition.value_kind is ProbeValueKind.NUMBER and not all(
        isinstance(item, (int, float)) and not isinstance(item, bool) for item in scalar_values
    ):
        raise ValueError(f"{label} expects numeric comparison values for {probe_id!r}.")
    if definition.value_kind is ProbeValueKind.BOOLEAN and not (
        isinstance(value, bool)
        or constraint.operator in {ConstraintOperator.REQUIRED, ConstraintOperator.PROHIBITED}
    ):
        raise ValueError(f"{label} expects a boolean comparison value for {probe_id!r}.")
    if definition.value_kind is ProbeValueKind.STRING and not all(
        isinstance(item, str) for item in scalar_values
    ):
        raise ValueError(f"{label} expects string comparison values for {probe_id!r}.")
    if definition.value_kind is ProbeValueKind.STRING_SET and not all(
        isinstance(item, str) for item in scalar_values
    ):
        raise ValueError(f"{label} expects string-set comparison values for {probe_id!r}.")
