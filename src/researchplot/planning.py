"""Export planning and coverage-aware project compliance semantics."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .compliance import Finding, Outcome, Policy, Report, TargetContext, Verdict
from .models import (
    ConstraintOperator,
    OutputFormat,
    RuleLevel,
    RulePhase,
    SourceRef,
    VenueProfile,
    VenueRule,
)
from .target import Target, coerce_format


@dataclass(frozen=True, slots=True)
class ExportSetting:
    """Resolved artifact settings for one selected output format."""

    format: OutputFormat
    minimum_dpi: float | None = None
    allowed_color_modes: tuple[str, ...] = ()
    compression: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "format": self.format.value,
            "minimum_dpi": self.minimum_dpi,
            "allowed_color_modes": list(self.allowed_color_modes),
            "compression": self.compression,
        }


@dataclass(frozen=True, slots=True)
class ExportPlan:
    """A no-write explanation of formats and settings selected for a target."""

    profile: str
    profile_digest: str
    target: TargetContext
    allowed_formats: tuple[OutputFormat, ...]
    selected_formats: tuple[OutputFormat, ...]
    preferred_format: OutputFormat | None
    required_companions: tuple[OutputFormat, ...]
    settings: tuple[ExportSetting, ...]
    format_unspecified: bool
    explanations: tuple[str, ...]

    def setting_for(self, output_format: OutputFormat | str) -> ExportSetting:
        selected = coerce_format(output_format)
        try:
            return next(item for item in self.settings if item.format is selected)
        except StopIteration as exc:
            choices = ", ".join(item.format.value for item in self.settings)
            raise KeyError(
                f"Format {selected.value!r} is not selected. Planned formats: {choices}."
            ) from exc

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "profile_digest": self.profile_digest,
            "target": self.target.to_dict(),
            "allowed_formats": [item.value for item in self.allowed_formats],
            "selected_formats": [item.value for item in self.selected_formats],
            "preferred_format": self.preferred_format.value
            if self.preferred_format is not None
            else None,
            "required_companions": [item.value for item in self.required_companions],
            "settings": [item.to_dict() for item in self.settings],
            "format_unspecified": self.format_unspecified,
            "explanations": list(self.explanations),
        }


def _rule_applies(
    rule: VenueRule, target: Target, output_format: OutputFormat | None = None
) -> bool:
    return rule.applies_to.matches(
        role=target.role,
        content_kind=target.content,
        output_format=output_format,
        width=target.width,
    )


def _rule_formats(rule: VenueRule) -> tuple[OutputFormat, ...]:
    if not isinstance(rule.value, tuple):
        return ()
    selected: list[OutputFormat] = []
    for item in rule.value:
        try:
            output_format = coerce_format(str(item))
        except ValueError:
            continue
        if output_format not in selected:
            selected.append(output_format)
    return tuple(selected)


def _allowed_formats(target: Target) -> tuple[OutputFormat, ...]:
    selected: list[OutputFormat] = []
    for rule in target.profile.rules:
        if not rule.id.startswith("export.formats.") or not _rule_applies(rule, target):
            continue
        for output_format in _rule_formats(rule):
            if output_format not in selected:
                selected.append(output_format)
    return tuple(selected)


def _required_companions(target: Target) -> tuple[OutputFormat, ...]:
    selected: list[OutputFormat] = []
    for rule in target.profile.rules:
        if (
            not rule.id.startswith("export.companions.")
            or rule.level is not RuleLevel.REQUIRED
            or not _rule_applies(rule, target)
        ):
            continue
        for output_format in _rule_formats(rule):
            if output_format not in selected:
                selected.append(output_format)
    return tuple(selected)


def _minimum_dpi(target: Target, output_format: OutputFormat) -> float | None:
    candidates: list[float] = []
    for rule in target.profile.rules:
        if rule.probe != "artifact.dpi" or not _rule_applies(rule, target, output_format):
            continue
        if (
            rule.constraint.operator
            in {ConstraintOperator.EQ, ConstraintOperator.GTE, ConstraintOperator.GT}
            and isinstance(rule.value, (int, float))
            and not isinstance(rule.value, bool)
        ):
            candidates.append(float(rule.value))
        elif (
            rule.constraint.operator is ConstraintOperator.BETWEEN
            and isinstance(rule.value, tuple)
            and len(rule.value) == 2
            and isinstance(rule.value[0], (int, float))
        ):
            candidates.append(float(rule.value[0]))
    return max(candidates) if candidates else None


def _allowed_modes(target: Target, output_format: OutputFormat) -> tuple[str, ...]:
    selected: list[str] = []
    for rule in target.profile.rules:
        if (
            rule.probe != "raster.mode"
            or rule.constraint.operator is not ConstraintOperator.IN
            or not _rule_applies(rule, target, output_format)
            or not isinstance(rule.value, tuple)
        ):
            continue
        for item in rule.value:
            name = str(item)
            if name not in selected:
                selected.append(name)
    return tuple(selected)


def _compression(target: Target, output_format: OutputFormat) -> str | None:
    for rule in target.profile.rules:
        if (
            rule.probe == "raster.compression"
            and rule.constraint.operator is ConstraintOperator.EQ
            and _rule_applies(rule, target, output_format)
            and isinstance(rule.value, str)
        ):
            return rule.value
    return None


def plan_export(
    target: Target,
    *,
    formats: Sequence[OutputFormat | str] | None = None,
    preferred: OutputFormat | str | None = None,
) -> ExportPlan:
    """Resolve an explicit, no-write export plan for a v1 or v2 target.

    Unlike the legacy implicit exporter, an omitted ``formats`` selects one preferred
    representation plus any explicitly encoded required companions.  This behavior is
    exposed through the planner only; :meth:`Target.export` retains its 1.x behavior.
    """

    allowed = _allowed_formats(target)
    companions = _required_companions(target)
    requested: list[OutputFormat] = []
    if formats is not None:
        for value in formats:
            selected = coerce_format(value)
            if selected not in requested:
                requested.append(selected)
        if not requested:
            raise ValueError("formats must contain at least one output format when provided.")
    selected_preference = coerce_format(preferred) if preferred is not None else None
    if selected_preference is not None and allowed and selected_preference not in allowed:
        choices = ", ".join(item.value for item in allowed)
        raise ValueError(
            f"Preferred format {selected_preference.value!r} is not allowed; choose from: {choices}."
        )
    if allowed:
        disallowed = [item for item in requested if item not in allowed]
        if disallowed:
            choices = ", ".join(item.value for item in allowed)
            rejected = ", ".join(item.value for item in disallowed)
            raise ValueError(
                f"Requested format(s) {rejected} are not allowed; choose from: {choices}."
            )
    if requested and selected_preference is not None and selected_preference not in requested:
        raise ValueError("preferred must be one of the explicitly requested formats.")

    explanations: list[str] = []
    if requested:
        selected_formats = list(requested)
        explanations.append("Selected the formats explicitly requested by the project.")
    else:
        chosen = selected_preference or (allowed[0] if allowed else None)
        selected_formats = [chosen] if chosen is not None else []
        if selected_preference is not None:
            explanations.append("Selected the caller's preferred allowed format.")
        elif chosen is not None:
            explanations.append(
                "Selected the first source-backed allowed format as a deterministic default; "
                "this ordering is not promoted to a venue requirement."
            )
        else:
            explanations.append(
                "The profile does not encode an allowed artifact format; choose one explicitly."
            )
    for companion in companions:
        if companion not in selected_formats:
            selected_formats.append(companion)
    if companions:
        explanations.append("Added source-backed required companion representations.")

    selected_tuple = tuple(selected_formats)
    settings = tuple(
        ExportSetting(
            output_format,
            minimum_dpi=_minimum_dpi(target, output_format),
            allowed_color_modes=_allowed_modes(target, output_format),
            compression=_compression(target, output_format),
        )
        for output_format in selected_tuple
    )
    return ExportPlan(
        profile=target.coordinate,
        profile_digest=target.profile.digest,
        target=target.context(),
        allowed_formats=allowed,
        selected_formats=selected_tuple,
        preferred_format=selected_preference or (allowed[0] if allowed else None),
        required_companions=companions,
        settings=settings,
        format_unspecified=not allowed,
        explanations=tuple(explanations),
    )


class CoverageStatus(StrEnum):
    """Whether required evidence was present and conclusive."""

    SATISFIED = "satisfied"
    FAILED = "failed"
    UNRESOLVED = "unresolved"
    MISSING = "missing"


@dataclass(frozen=True, slots=True)
class CoverageRequirement:
    """One required rule/phase that a project plan expects evidence for."""

    figure_id: str
    rule_id: str
    phases: tuple[RulePhase, ...]
    deliverable_id: str | None = None

    @property
    def key(self) -> tuple[str, str | None, str]:
        return self.figure_id, self.deliverable_id, self.rule_id

    def to_dict(self) -> dict[str, object]:
        return {
            "figure_id": self.figure_id,
            "deliverable_id": self.deliverable_id,
            "rule_id": self.rule_id,
            "phases": [item.value for item in self.phases],
        }


@dataclass(frozen=True, slots=True)
class PlanEvidence:
    """A report associated with one logical figure and optional deliverable."""

    figure_id: str
    report: Report
    deliverable_id: str | None = None


@dataclass(frozen=True, slots=True)
class CoverageResult:
    requirement: CoverageRequirement
    status: CoverageStatus
    finding: Finding | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "requirement": self.requirement.to_dict(),
            "status": self.status.value,
            "finding": self.finding.to_dict() if self.finding is not None else None,
        }


@dataclass(frozen=True, slots=True)
class FigurePlan:
    figure_id: str
    target: Target
    export: ExportPlan
    deliverable_ids: tuple[str, ...]
    waiver_rule_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "figure_id": self.figure_id,
            "target": self.target.context().to_dict(),
            "export": self.export.to_dict(),
            "deliverable_ids": list(self.deliverable_ids),
            "waiver_rule_ids": list(self.waiver_rule_ids),
        }


@dataclass(frozen=True, slots=True)
class PlanAssessment:
    """Coverage-aware project verdict and all evidence used to derive it."""

    profile: str
    profile_digest: str
    evidence: tuple[PlanEvidence, ...]
    coverage: tuple[CoverageResult, ...]
    capability_gaps: tuple[str, ...] = ()
    plan_digest: str = ""
    environment_provenance: tuple[tuple[str, str], ...] = ()
    schema_version: int = 2

    @property
    def findings(self) -> tuple[Finding, ...]:
        return tuple(finding for item in self.evidence for finding in item.report.findings)

    @property
    def verdict(self) -> Verdict:
        if any(
            finding.level is RuleLevel.REQUIRED and finding.outcome is Outcome.FAIL
            for finding in self.findings
        ):
            return Verdict.NON_COMPLIANT
        if self.capability_gaps or any(
            item.status in {CoverageStatus.MISSING, CoverageStatus.UNRESOLVED}
            for item in self.coverage
        ):
            return Verdict.INDETERMINATE
        if any(
            finding.level is RuleLevel.REQUIRED and finding.outcome is Outcome.SKIP
            for finding in self.findings
        ):
            return Verdict.INDETERMINATE
        return Verdict.COMPLIANT

    @property
    def passed(self) -> bool:
        return self.verdict is Verdict.COMPLIANT

    def blocks(self, policy: Policy | str = Policy.COMPLETE) -> bool:
        """Return whether a v1-compatible policy blocks this aggregate verdict."""

        selected = Policy(policy)
        if selected is Policy.OFF:
            return False
        if selected is Policy.VIOLATIONS:
            return self.verdict is Verdict.NON_COMPLIANT
        return self.verdict is not Verdict.COMPLIANT

    @property
    def failures(self) -> tuple[Finding, ...]:
        return tuple(item for item in self.findings if item.outcome is Outcome.FAIL)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        """Recommendation failures that do not determine the venue verdict."""

        return tuple(
            item
            for item in self.findings
            if item.level is RuleLevel.RECOMMENDED and item.outcome is Outcome.FAIL
        )

    @property
    def sources(self) -> tuple[SourceRef, ...]:
        """Unique official sources carried by all phase reports."""

        unique: dict[str, SourceRef] = {}
        for item in self.evidence:
            for source in item.report.sources:
                unique.setdefault(source.id, source)
        return tuple(unique.values())

    @property
    def remediations(self) -> tuple[str, ...]:
        """Unique deterministic suggestions attached to evaluated findings."""

        return tuple(
            dict.fromkeys(
                item.suggestion.strip()
                for item in self.findings
                if item.suggestion is not None and item.suggestion.strip()
            )
        )

    @property
    def unresolved(self) -> tuple[CoverageResult, ...]:
        return tuple(
            item
            for item in self.coverage
            if item.status in {CoverageStatus.MISSING, CoverageStatus.UNRESOLVED}
        )

    def to_dict(self) -> dict[str, object]:
        phase_coverage: list[dict[str, object]] = []
        for phase in RulePhase:
            phase_results = tuple(
                item for item in self.coverage if phase in item.requirement.phases
            )
            phase_coverage.append(
                {
                    "phase": phase.value,
                    "applicable_required": len(phase_results),
                    "satisfied": sum(
                        item.status is CoverageStatus.SATISFIED for item in phase_results
                    ),
                    "failed": sum(item.status is CoverageStatus.FAILED for item in phase_results),
                    "unresolved": sum(
                        item.status in {CoverageStatus.MISSING, CoverageStatus.UNRESOLVED}
                        for item in phase_results
                    ),
                }
            )
        unresolved_checks: list[dict[str, object]] = [
            {
                "kind": "coverage",
                "figure_id": item.requirement.figure_id,
                "deliverable_id": item.requirement.deliverable_id,
                "rule_id": item.requirement.rule_id,
                "phases": [phase.value for phase in item.requirement.phases],
                "status": item.status.value,
            }
            for item in self.unresolved
        ]
        unresolved_checks.extend(
            {"kind": "capability", "message": message} for message in self.capability_gaps
        )
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "profile_digest": self.profile_digest,
            "plan_digest": self.plan_digest,
            "verdict": self.verdict.value,
            "summary": {
                "findings": len(self.findings),
                "failures": len(self.failures),
                "warnings": len(self.warnings),
                "unresolved": len(self.unresolved) + len(self.capability_gaps),
            },
            "sources": [source.to_dict() for source in self.sources],
            "findings": [finding.to_dict() for finding in self.findings],
            "observations": [
                {
                    "rule_id": finding.rule_id,
                    "phase": finding.phase,
                    "artifact": finding.artifact,
                    "observed": finding.observed,
                    "expected": finding.expected,
                    "outcome": finding.outcome.value,
                }
                for finding in self.findings
            ],
            "remediations": list(self.remediations),
            "phase_coverage": phase_coverage,
            "coverage": [item.to_dict() for item in self.coverage],
            "unresolved_checks": unresolved_checks,
            "capability_gaps": list(self.capability_gaps),
            "environment_provenance": dict(self.environment_provenance),
            "evidence": [
                {
                    "figure_id": item.figure_id,
                    "deliverable_id": item.deliverable_id,
                    "report": item.report.to_dict(),
                }
                for item in self.evidence
            ],
        }


@dataclass(frozen=True, slots=True)
class CompliancePlan:
    """Resolved project targets plus explicit required-evidence coverage."""

    profile: VenueProfile
    figures: tuple[FigurePlan, ...]
    requirements: tuple[CoverageRequirement, ...]
    capability_gaps: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ids = [item.figure_id for item in self.figures]
        if len(ids) != len(set(ids)):
            raise ValueError("CompliancePlan figure ids must be unique.")
        keys = [item.key for item in self.requirements]
        if len(keys) != len(set(keys)):
            raise ValueError("CompliancePlan coverage requirements must be unique.")
        if any(not item.strip() for item in self.capability_gaps):
            raise ValueError("CompliancePlan capability gaps must be non-empty strings.")
        if len(self.capability_gaps) != len(set(self.capability_gaps)):
            raise ValueError("CompliancePlan capability gaps must be unique.")

    def assess(self, evidence: Iterable[PlanEvidence]) -> PlanAssessment:
        """Derive a tri-state verdict without treating absent phases as success."""

        values = tuple(evidence)
        figures = {item.figure_id: set(item.deliverable_ids) for item in self.figures}
        for item in values:
            if item.figure_id not in figures:
                raise ValueError(f"Evidence references unknown figure {item.figure_id!r}.")
            if (
                item.deliverable_id is not None
                and item.deliverable_id not in figures[item.figure_id]
            ):
                raise ValueError(
                    f"Evidence references unknown deliverable {item.figure_id}/{item.deliverable_id}."
                )
            if item.report.profile != self.profile.coordinate:
                raise ValueError(
                    f"Evidence profile {item.report.profile!r} does not match {self.profile.coordinate!r}."
                )
            if item.report.profile_digest and item.report.profile_digest != self.profile.digest:
                raise ValueError("Evidence profile digest does not match the compliance plan.")

        priority = {Outcome.PASS: 0, Outcome.SKIP: 1, Outcome.FAIL: 2}
        coverage: list[CoverageResult] = []
        for requirement in self.requirements:
            candidates: list[Finding] = []
            for item in values:
                if item.figure_id != requirement.figure_id:
                    continue
                if item.deliverable_id != requirement.deliverable_id:
                    continue
                candidates.extend(
                    finding
                    for finding in item.report.findings
                    if finding.rule_id == requirement.rule_id
                    and finding.phase in {phase.value for phase in requirement.phases}
                )
            if not candidates:
                coverage.append(CoverageResult(requirement, CoverageStatus.MISSING))
                continue
            strongest = max(candidates, key=lambda item: priority[item.outcome])
            status = {
                Outcome.PASS: CoverageStatus.SATISFIED,
                Outcome.FAIL: CoverageStatus.FAILED,
                Outcome.SKIP: CoverageStatus.UNRESOLVED,
            }[strongest.outcome]
            coverage.append(CoverageResult(requirement, status, strongest))
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        plan_digest = hashlib.sha256(payload).hexdigest()
        from .provenance import collect_environment_provenance

        return PlanAssessment(
            self.profile.coordinate,
            self.profile.digest,
            values,
            tuple(coverage),
            self.capability_gaps,
            plan_digest,
            collect_environment_provenance(),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile.coordinate,
            "profile_digest": self.profile.digest,
            "figures": [item.to_dict() for item in self.figures],
            "requirements": [item.to_dict() for item in self.requirements],
            "capability_gaps": list(self.capability_gaps),
        }
