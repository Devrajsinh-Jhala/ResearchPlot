"""Public immutable models used by ResearchPlot.

The profile models deliberately contain no Matplotlib objects.  They can be
loaded, inspected, hashed, and serialised in a completely offline process.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

RuleValue: TypeAlias = str | int | float | bool | tuple[str, ...] | tuple[float, ...] | None


class VenueKind(StrEnum):
    """The publication venue category."""

    JOURNAL = "journal"
    CONFERENCE = "conference"
    PUBLISHER = "publisher"


class RuleLevel(StrEnum):
    """How authoritative a venue rule is."""

    REQUIRED = "required"
    RECOMMENDED = "recommended"
    INFERRED = "inferred"


class FigureRole(StrEnum):
    """Where a figure will appear in a submission."""

    MAIN = "main"
    EXTENDED_DATA = "extended_data"
    SUPPLEMENTARY = "supplementary"
    GRAPHICAL_ABSTRACT = "graphical_abstract"


class ContentKind(StrEnum):
    """The visual content represented by an artifact.

    This is intentionally separate from :class:`OutputFormat`: a line-art
    figure may be exported as either PDF or TIFF, for example.
    """

    DATA_VISUALIZATION = "data_visualization"
    LINE_ART = "line_art"
    HALFTONE = "halftone"
    COMBINATION = "combination"
    PHOTOGRAPH = "photograph"


class OutputFormat(StrEnum):
    """File formats understood by bundled profile applicability rules."""

    PDF = "pdf"
    EPS = "eps"
    SVG = "svg"
    PNG = "png"
    JPEG = "jpeg"
    TIFF = "tiff"


class VerificationMode(StrEnum):
    """How ResearchPlot can establish whether a rule is satisfied."""

    AUTOMATED = "automated"
    MANUAL = "manual"
    UNSUPPORTED = "unsupported"


class RulePhase(StrEnum):
    """The validation stage in which a rule can be evaluated."""

    LIVE = "live"
    FILE = "file"
    BUNDLE = "bundle"


class ConstraintOperator(StrEnum):
    """A stable, machine-readable comparison operator."""

    EQ = "eq"
    NE = "ne"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    SUBSET = "subset"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    APPROX = "approx"
    REQUIRED = "required"
    PROHIBITED = "prohibited"


@dataclass(frozen=True, slots=True)
class RuleConstraint:
    """The comparison encoded by a venue rule."""

    operator: ConstraintOperator
    value: RuleValue
    unit: str | None = None
    tolerance: float | None = None

    def to_dict(self) -> dict[str, object]:
        value: object = list(self.value) if isinstance(self.value, tuple) else self.value
        return {
            "operator": self.operator.value,
            "value": value,
            "unit": self.unit,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True, slots=True)
class RuleApplicability:
    """Conditions under which a rule is relevant.

    An empty tuple means "all values" for that dimension.  This makes the
    common, venue-wide rule compact while keeping applicability explicit for
    role-, content-, and format-specific requirements.
    """

    roles: tuple[FigureRole, ...] = ()
    content_kinds: tuple[ContentKind, ...] = ()
    output_formats: tuple[OutputFormat, ...] = ()
    widths: tuple[str, ...] = ()

    @property
    def formats(self) -> tuple[OutputFormat, ...]:
        """Short compatibility alias for :attr:`output_formats`."""

        return self.output_formats

    def matches(
        self,
        *,
        role: FigureRole | str | None = None,
        content_kind: ContentKind | str | None = None,
        output_format: OutputFormat | str | None = None,
        width: str | None = None,
    ) -> bool:
        """Return whether supplied target metadata satisfies this filter."""

        selected_role = FigureRole(role) if isinstance(role, str) else role
        selected_content = (
            ContentKind(content_kind) if isinstance(content_kind, str) else content_kind
        )
        selected_format = (
            OutputFormat(output_format) if isinstance(output_format, str) else output_format
        )
        return (
            (not self.roles or selected_role in self.roles)
            and (not self.content_kinds or selected_content in self.content_kinds)
            and (not self.output_formats or selected_format in self.output_formats)
            and (not self.widths or width in self.widths)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "roles": [item.value for item in self.roles],
            "content_kinds": [item.value for item in self.content_kinds],
            "output_formats": [item.value for item in self.output_formats],
            "widths": list(self.widths),
        }


@dataclass(frozen=True, slots=True)
class SourceRef:
    """An official source used to establish venue rules."""

    id: str
    title: str
    url: str
    locator: str
    retrieved_on: str
    verified_on: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "locator": self.locator,
            "retrieved_on": self.retrieved_on,
            "verified_on": self.verified_on,
        }


@dataclass(frozen=True, slots=True)
class VenueRule:
    """A single source-backed rule in a venue profile."""

    id: str
    probe: str
    constraint: RuleConstraint
    applies_to: RuleApplicability
    verification: VerificationMode
    level: RuleLevel
    source_ids: tuple[str, ...]
    description: str
    phases: tuple[RulePhase, ...] = (RulePhase.LIVE, RulePhase.FILE)

    @property
    def value(self) -> RuleValue:
        """Compatibility view of :attr:`constraint` for the 0.2 API."""

        return self.constraint.value

    @property
    def unit(self) -> str | None:
        """Compatibility view of :attr:`constraint` for the 0.2 API."""

        return self.constraint.unit

    @property
    def applicability(self) -> RuleApplicability:
        """Readable alias for the JSON field name :attr:`applies_to`."""

        return self.applies_to

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "probe": self.probe,
            "constraint": self.constraint.to_dict(),
            "applies_to": self.applies_to.to_dict(),
            "verification": self.verification.value,
            "phases": [phase.value for phase in self.phases],
            "level": self.level.value,
            "source_ids": list(self.source_ids),
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class VenueProfile:
    """Immutable, validated venue specification."""

    id: str
    name: str
    kind: VenueKind
    year: int | None
    aliases: tuple[str, ...]
    scope: str
    default_width: str | None
    verified_on: str
    sources: tuple[SourceRef, ...]
    rules: tuple[VenueRule, ...]
    caveats: tuple[str, ...] = ()
    schema_version: int = 2
    revision: str = "unversioned"
    effective_date: str = "1970-01-01"
    digest: str = ""

    @property
    def coordinate(self) -> str:
        """Immutable profile coordinate, for example ``nature@2026.08.0``."""

        return f"{self.id}@{self.revision}"

    @property
    def profile_revision(self) -> str:
        """Explicit alias used by profile-lock and manifest consumers."""

        return self.revision

    def get_rule(self, rule_id: str) -> VenueRule | None:
        """Return a rule by identifier, or ``None`` when unspecified."""

        return next((rule for rule in self.rules if rule.id == rule_id), None)

    def rules_with_prefix(self, prefix: str) -> tuple[VenueRule, ...]:
        """Return all rules whose identifiers start with ``prefix``."""

        return tuple(rule for rule in self.rules if rule.id.startswith(prefix))

    @property
    def width_options(self) -> tuple[str, ...]:
        """Available figure width names for this profile."""

        prefix = "figure.width."
        return tuple(
            rule.id.removeprefix(prefix)
            for rule in self.rules_with_prefix(prefix)
            if rule.constraint.operator in {ConstraintOperator.EQ, ConstraintOperator.APPROX}
            and isinstance(rule.value, (int, float))
            and not isinstance(rule.value, bool)
        )

    def width_mm(self, width: str | None = None) -> float:
        """Return an allowed figure width in millimetres."""

        selected = self.default_width if width is None else width
        if selected is None:
            raise ValueError(
                f"Profile {self.coordinate!r} does not specify physical figure widths."
            )
        if not isinstance(selected, str):
            raise TypeError("Figure width must be a string or null.")
        if not selected.strip():
            raise ValueError("Figure width must be a non-empty string.")
        rule = self.get_rule(f"figure.width.{selected}")
        if rule is None or not isinstance(rule.value, (int, float)):
            choices = ", ".join(self.width_options)
            raise ValueError(
                f"Unknown width {selected!r} for {self.id}. Available widths: {choices}."
            )
        return float(rule.value)

    def source_urls_for(self, rule: VenueRule) -> tuple[str, ...]:
        source_by_id = {source.id: source for source in self.sources}
        return tuple(
            source_by_id[source_id].url
            for source_id in rule.source_ids
            if source_id in source_by_id
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "coordinate": self.coordinate,
            "schema_version": self.schema_version,
            "revision": self.revision,
            "effective_date": self.effective_date,
            "digest": self.digest,
            "name": self.name,
            "kind": self.kind.value,
            "year": self.year,
            "aliases": list(self.aliases),
            "scope": self.scope,
            "default_width": self.default_width,
            "verified_on": self.verified_on,
            "sources": [source.to_dict() for source in self.sources],
            "rules": [rule.to_dict() for rule in self.rules],
            "caveats": list(self.caveats),
            "width_options": list(self.width_options),
        }


class VenueResolutionWarning(UserWarning):
    """A query resolved to a versioned or caveated venue profile."""
