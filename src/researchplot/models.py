"""Public immutable models used by ResearchPlot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TypeAlias

RuleValue: TypeAlias = str | int | float | bool | tuple[str, ...] | tuple[float, ...] | None


class VenueKind(str, Enum):
    """The publication venue category."""

    JOURNAL = "journal"
    CONFERENCE = "conference"
    PUBLISHER = "publisher"


class RuleLevel(str, Enum):
    """How authoritative a venue rule is."""

    REQUIRED = "required"
    RECOMMENDED = "recommended"
    INFERRED = "inferred"


class CheckStatus(str, Enum):
    """Result status for a validation check."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    INFO = "info"
    SKIP = "skip"


class ArtworkType(str, Enum):
    """Artwork categories used by publication guidelines."""

    VECTOR = "vector"
    HALFTONE = "halftone"
    COMBINATION = "combination"
    LINE_ART = "line_art"


@dataclass(frozen=True, slots=True)
class SourceRef:
    """An official source used to establish venue rules."""

    id: str
    title: str
    url: str
    verified_on: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "verified_on": self.verified_on,
        }


@dataclass(frozen=True, slots=True)
class VenueRule:
    """A single source-backed rule in a venue profile."""

    id: str
    value: RuleValue
    unit: str | None
    level: RuleLevel
    source_ids: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, object]:
        value: object = list(self.value) if isinstance(self.value, tuple) else self.value
        return {
            "id": self.id,
            "value": value,
            "unit": self.unit,
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
    default_width: str
    verified_on: str
    sources: tuple[SourceRef, ...]
    rules: tuple[VenueRule, ...]
    caveats: tuple[str, ...] = ()

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
        return tuple(rule.id.removeprefix(prefix) for rule in self.rules_with_prefix(prefix))

    def width_mm(self, width: str | None = None) -> float:
        """Return an allowed figure width in millimetres."""

        selected = width or self.default_width
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


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One measured compliance check."""

    rule_id: str
    status: CheckStatus
    level: RuleLevel
    message: str
    observed: object = None
    expected: object = None
    source_urls: tuple[str, ...] = ()
    suggestion: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "status": self.status.value,
            "level": self.level.value,
            "message": self.message,
            "observed": self.observed,
            "expected": self.expected,
            "source_urls": list(self.source_urls),
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """A serialisable collection of compliance checks."""

    profile_id: str
    width: str
    artwork: ArtworkType
    checks: tuple[CheckResult, ...]
    generated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    @property
    def passed(self) -> bool:
        return not self.failures

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if check.status is CheckStatus.FAIL)

    @property
    def warnings(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if check.status is CheckStatus.WARN)

    def to_dict(self) -> dict[str, object]:
        return {
            "profile_id": self.profile_id,
            "width": self.width,
            "artwork": self.artwork.value,
            "generated_at": self.generated_at,
            "passed": self.passed,
            "checks": [check.to_dict() for check in self.checks],
        }

    def __str__(self) -> str:
        rows = [f"ResearchPlot validation: {self.profile_id} ({self.width})"]
        for check in self.checks:
            rows.append(f"[{check.status.value.upper():4}] {check.message}")
        rows.append(
            f"{len(self.checks)} checks; {len(self.failures)} failure(s); "
            f"{len(self.warnings)} warning(s)"
        )
        return "\n".join(rows)


class VenueResolutionWarning(UserWarning):
    """A query resolved to a versioned or caveated venue profile."""


class LegacyStyleWarning(FutureWarning):
    """A legacy plotting style is not a verified compliance profile."""


class ComplianceError(ValueError):
    """Raised when strict export is blocked by required-rule failures."""

    def __init__(self, report: ValidationReport) -> None:
        self.report = report
        super().__init__(
            f"Export blocked: {len(report.failures)} required-rule failure(s) for "
            f"{report.profile_id}."
        )
