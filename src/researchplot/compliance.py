"""Generic profile-rule evaluation and stable compliance reports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum

from .models import (
    ConstraintOperator,
    RuleConstraint,
    RuleLevel,
    RulePhase,
    SourceRef,
    VenueProfile,
    VenueRule,
)
from .observations import ObservationSet

REPORT_SCHEMA_VERSION = 1


class Outcome(StrEnum):
    """Whether an applicable rule passed, failed, or could not be established."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


class Verdict(StrEnum):
    """Overall compliance result for required rules."""

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    INDETERMINATE = "indeterminate"


class Policy(StrEnum):
    """Blocking policy used by export and CI workflows."""

    VIOLATIONS = "violations"
    COMPLETE = "complete"
    OFF = "off"


@dataclass(frozen=True, slots=True)
class TargetContext:
    """Metadata used to select conditional profile rules."""

    role: str
    width: str | None
    content: str
    output_format: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "width": self.width,
            "content": self.content,
            "output_format": self.output_format,
        }


@dataclass(frozen=True, slots=True)
class Finding:
    """One rule evaluation with measured and source-backed evidence."""

    rule_id: str
    outcome: Outcome
    level: RuleLevel
    phase: str
    message: str
    observed: object = None
    expected: object = None
    source_urls: tuple[str, ...] = ()
    suggestion: str | None = None
    verification: str = "automated"
    sources: tuple[SourceRef, ...] = ()
    artifact: str | None = None

    @property
    def blocking_failure(self) -> bool:
        return self.level is RuleLevel.REQUIRED and self.outcome is Outcome.FAIL

    @property
    def unresolved_required(self) -> bool:
        return self.level is RuleLevel.REQUIRED and self.outcome is Outcome.SKIP

    def to_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "outcome": self.outcome.value,
            "level": self.level.value,
            "phase": self.phase,
            "verification": self.verification,
            "message": self.message,
            "observed": self.observed,
            "expected": self.expected,
            "source_urls": list(self.source_urls),
            "sources": [source.to_dict() for source in self.sources],
            "artifact": self.artifact,
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True, slots=True)
class Report:
    """Deterministic, serialisable compliance report."""

    profile: str
    target: TargetContext
    findings: tuple[Finding, ...]
    schema_version: int = REPORT_SCHEMA_VERSION
    profile_digest: str = ""
    sources: tuple[SourceRef, ...] = ()
    caveats: tuple[str, ...] = ()

    @property
    def verdict(self) -> Verdict:
        if any(item.blocking_failure for item in self.findings):
            return Verdict.NON_COMPLIANT
        if any(item.unresolved_required for item in self.findings):
            return Verdict.INDETERMINATE
        return Verdict.COMPLIANT

    @property
    def passed(self) -> bool:
        """Compatibility spelling; true only for a complete compliant report."""

        return self.verdict is Verdict.COMPLIANT

    @property
    def failures(self) -> tuple[Finding, ...]:
        return tuple(item for item in self.findings if item.outcome is Outcome.FAIL)

    @property
    def warnings(self) -> tuple[Finding, ...]:
        return tuple(
            item
            for item in self.findings
            if item.level is RuleLevel.RECOMMENDED and item.outcome is Outcome.FAIL
        )

    @property
    def unresolved(self) -> tuple[Finding, ...]:
        return tuple(item for item in self.findings if item.outcome is Outcome.SKIP)

    def blocks(self, policy: Policy | str = Policy.COMPLETE) -> bool:
        selected = Policy(policy)
        if selected is Policy.OFF:
            return False
        if selected is Policy.VIOLATIONS:
            return self.verdict is Verdict.NON_COMPLIANT
        return self.verdict is not Verdict.COMPLIANT

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "profile": self.profile,
            "profile_digest": self.profile_digest,
            "sources": [source.to_dict() for source in self.sources],
            "caveats": list(self.caveats),
            "target": self.target.to_dict(),
            "verdict": self.verdict.value,
            "summary": {
                "findings": len(self.findings),
                "failures": len(self.failures),
                "warnings": len(self.warnings),
                "unresolved": len(self.unresolved),
            },
            "findings": [item.to_dict() for item in self.findings],
        }

    def __str__(self) -> str:
        rows = [f"ResearchPlot: {self.verdict.value} — {self.profile}"]
        icons = {Outcome.PASS: "PASS", Outcome.FAIL: "FAIL", Outcome.SKIP: "SKIP"}
        rows.extend(f"[{icons[item.outcome]:4}] {item.message}" for item in self.findings)
        rows.append(
            f"{len(self.findings)} findings; {len(self.failures)} failed; "
            f"{len(self.unresolved)} unresolved"
        )
        return "\n".join(rows)

    @classmethod
    def combine(cls, reports: Iterable[Report]) -> Report:
        values = tuple(reports)
        if not values:
            raise ValueError("At least one report is required.")
        first = values[0]
        if any(
            item.profile != first.profile
            or item.profile_digest != first.profile_digest
            or item.target != first.target
            for item in values[1:]
        ):
            raise ValueError("Reports can be combined only when their profile and target match.")
        unique: dict[tuple[str, str, str | None], Finding] = {}
        outcome_priority = {Outcome.PASS: 0, Outcome.SKIP: 1, Outcome.FAIL: 2}
        for report in values:
            for finding in report.findings:
                key = (finding.phase, finding.rule_id, finding.artifact)
                previous = unique.get(key)
                if (
                    previous is None
                    or outcome_priority[finding.outcome] > outcome_priority[previous.outcome]
                ):
                    unique[key] = finding
        return cls(
            first.profile,
            first.target,
            tuple(unique.values()),
            schema_version=first.schema_version,
            profile_digest=first.profile_digest,
            sources=first.sources,
            caveats=first.caveats,
        )


class CompliancePolicyError(ValueError):
    """Raised when a report is blocked by the requested compliance policy."""

    def __init__(self, report: Report, policy: Policy | str) -> None:
        self.report = report
        self.policy = Policy(policy)
        super().__init__(
            f"{self.policy.value} policy blocked {report.profile}: {report.verdict.value}."
        )


def _applies(rule: VenueRule, target: TargetContext, phase: RulePhase) -> bool:
    selected_phase = phase
    if selected_phase not in rule.phases:
        return False
    return rule.applies_to.matches(
        role=target.role,
        width=target.width,
        content_kind=target.content,
        output_format=target.output_format,
    )


def _operator_name(constraint: RuleConstraint) -> str:
    return constraint.operator.value


def _expected(constraint: RuleConstraint) -> object:
    return constraint.value


def _number(value: object, *, label: str) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    raise TypeError(f"{label} must be numeric, not {type(value).__name__}.")


def _sequence(value: object, *, label: str) -> Sequence[object]:
    if isinstance(value, (str, tuple, list)):
        return value
    raise TypeError(f"{label} must be a sequence, not {type(value).__name__}.")


def _evaluate_constraint(observed: object, constraint: RuleConstraint) -> bool:
    operator = constraint.operator
    expected = constraint.value
    if operator is ConstraintOperator.EQ:
        return observed == expected
    if operator is ConstraintOperator.NE:
        return observed != expected
    if operator in {
        ConstraintOperator.GT,
        ConstraintOperator.GTE,
        ConstraintOperator.LT,
        ConstraintOperator.LTE,
    }:
        actual_number = _number(observed, label="observed value")
        expected_number = _number(expected, label="expected value")
        if operator is ConstraintOperator.GT:
            return actual_number > expected_number
        if operator is ConstraintOperator.GTE:
            return actual_number >= expected_number
        if operator is ConstraintOperator.LT:
            return actual_number < expected_number
        return actual_number <= expected_number
    if operator is ConstraintOperator.IN:
        return observed in _sequence(expected, label="expected value")
    if operator is ConstraintOperator.NOT_IN:
        return observed not in _sequence(expected, label="expected value")
    if operator is ConstraintOperator.CONTAINS:
        return expected in _sequence(observed, label="observed value")
    if operator is ConstraintOperator.NOT_CONTAINS:
        return expected not in _sequence(observed, label="observed value")
    if operator is ConstraintOperator.SUBSET:
        observed_values = _sequence(observed, label="observed value")
        expected_values = _sequence(expected, label="expected value")
        return all(item in expected_values for item in observed_values)
    if operator is ConstraintOperator.BETWEEN:
        bounds = _sequence(expected, label="expected value")
        if len(bounds) != 2:
            raise TypeError("between requires exactly two numeric bounds.")
        actual_number = _number(observed, label="observed value")
        return (
            _number(bounds[0], label="lower bound")
            <= actual_number
            <= _number(bounds[1], label="upper bound")
        )
    if operator is ConstraintOperator.APPROX:
        allowed = constraint.tolerance if constraint.tolerance is not None else 0.5
        return (
            abs(
                _number(observed, label="observed value")
                - _number(expected, label="expected value")
            )
            <= allowed
        )
    if operator is ConstraintOperator.REQUIRED:
        return bool(observed)
    if operator is ConstraintOperator.PROHIBITED:
        return not bool(observed) if expected is True else observed != expected
    raise ValueError(f"Unsupported constraint operator {operator.value!r}.")


def _profile_coordinate(profile: VenueProfile) -> str:
    return str(getattr(profile, "coordinate", profile.id))


class RuleEngine:
    """Evaluate applicable profile rules against typed observations."""

    def evaluate(
        self,
        profile: VenueProfile,
        observations: ObservationSet,
        target: TargetContext,
        *,
        phase: str,
        attestations: Mapping[str, str] | None = None,
    ) -> Report:
        findings: list[Finding] = []
        attestations = attestations or {}
        selected_phase = RulePhase(phase)
        for rule in profile.rules:
            if not _applies(rule, target, selected_phase):
                continue
            verification = rule.verification.value
            source_urls = profile.source_urls_for(rule)
            source_ids = set(rule.source_ids)
            sources = tuple(source for source in profile.sources if source.id in source_ids)
            if verification == "unsupported":
                findings.append(
                    Finding(
                        rule.id,
                        Outcome.SKIP,
                        rule.level,
                        phase,
                        f"Rule {rule.id} is recorded but has no supported verification method.",
                        observed=None,
                        expected=rule.description,
                        source_urls=source_urls,
                        verification=verification,
                        sources=sources,
                    )
                )
                continue
            if verification == "manual":
                raw_attestation = attestations.get(rule.id)
                attestation = (
                    raw_attestation.strip()
                    if isinstance(raw_attestation, str) and raw_attestation.strip()
                    else None
                )
                findings.append(
                    Finding(
                        rule.id,
                        Outcome.PASS if attestation else Outcome.SKIP,
                        rule.level,
                        phase,
                        (
                            f"Manual requirement {rule.id} was attested."
                            if attestation
                            else f"Manual requirement {rule.id} requires an attestation."
                        ),
                        observed=attestation,
                        expected=rule.description,
                        source_urls=source_urls,
                        verification=verification,
                        sources=sources,
                    )
                )
                continue
            probe = rule.probe
            observation = observations.get(probe)
            phase_mismatch = observation is not None and observation.phase != selected_phase.value
            if observation is None or not observation.available or phase_mismatch:
                findings.append(
                    Finding(
                        rule.id,
                        Outcome.SKIP,
                        rule.level,
                        phase,
                        (
                            f"Probe {probe!r} was observed during {observation.phase!r}, "
                            f"not {selected_phase.value!r}."
                            if phase_mismatch and observation is not None
                            else observation.detail
                            if observation is not None and observation.detail
                            else f"Probe {probe!r} is unavailable."
                        ),
                        observed=None,
                        expected=_expected(rule.constraint),
                        source_urls=source_urls,
                        verification=verification,
                        sources=sources,
                    )
                )
                continue
            constraint = rule.constraint
            try:
                passed = _evaluate_constraint(observation.value, constraint)
            except (TypeError, ValueError) as exc:
                findings.append(
                    Finding(
                        rule.id,
                        Outcome.SKIP,
                        rule.level,
                        phase,
                        f"Could not evaluate {probe}: {exc}",
                        observed=observation.value,
                        expected=_expected(constraint),
                        source_urls=source_urls,
                        verification=verification,
                        sources=sources,
                    )
                )
                continue
            findings.append(
                Finding(
                    rule.id,
                    Outcome.PASS if passed else Outcome.FAIL,
                    rule.level,
                    phase,
                    (
                        f"{rule.description}"
                        if passed
                        else f"{rule.description} Observed value does not satisfy the rule."
                    ),
                    observed=observation.value,
                    expected=_expected(constraint),
                    source_urls=source_urls,
                    suggestion=None
                    if passed
                    else f"Adjust {probe} to satisfy {_operator_name(constraint)} {_expected(constraint)!r}.",
                    verification=verification,
                    sources=sources,
                )
            )
        return Report(
            _profile_coordinate(profile),
            target,
            tuple(findings),
            profile_digest=profile.digest,
            sources=profile.sources,
            caveats=profile.caveats,
        )
