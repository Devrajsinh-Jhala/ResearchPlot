"""Governance checks for source-backed venue profiles."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .models import ProfileStatus, RuleLevel, VenueProfile, VerificationMode
from .probes import validate_probe_constraint


@dataclass(frozen=True, slots=True)
class GovernanceIssue:
    code: str
    message: str
    error: bool = True


@dataclass(frozen=True, slots=True)
class GovernanceReport:
    issues: tuple[GovernanceIssue, ...]

    @property
    def passed(self) -> bool:
        return not any(issue.error for issue in self.issues)

    @property
    def errors(self) -> tuple[GovernanceIssue, ...]:
        return tuple(issue for issue in self.issues if issue.error)

    @property
    def warnings(self) -> tuple[GovernanceIssue, ...]:
        return tuple(issue for issue in self.issues if not issue.error)


def validate_profile_governance(profile: VenueProfile) -> GovernanceReport:
    """Validate evidence metadata and the closed probe/unit vocabulary."""

    issues: list[GovernanceIssue] = []
    if profile.status is ProfileStatus.VERIFIED:
        if not profile.maintainers:
            issues.append(
                GovernanceIssue("governance.maintainers", "Verified profiles need a maintainer.")
            )
        if not profile.governance.reviewers:
            issues.append(
                GovernanceIssue("governance.reviewers", "Verified profiles need a reviewer.")
            )
        if profile.governance.reviewed_on is None:
            issues.append(
                GovernanceIssue("governance.reviewed_on", "Verified profiles need a review date.")
            )
    if profile.status is ProfileStatus.TRANSLATED:
        issues.append(
            GovernanceIssue(
                "governance.translated",
                "Schema-v2 evidence metadata was synthesized during translation.",
                False,
            )
        )

    source_ids = {source.id for source in profile.sources}
    for source in profile.sources:
        if source.content_sha256 is not None and not re.fullmatch(
            r"[0-9a-f]{64}", source.content_sha256
        ):
            issues.append(
                GovernanceIssue(
                    "source.digest", f"Source {source.id!r} has an invalid SHA-256 digest."
                )
            )
        if date.fromisoformat(source.verified_on) > date.fromisoformat(profile.verified_on):
            issues.append(
                GovernanceIssue(
                    "source.verified_after_profile",
                    f"Source {source.id!r} was verified after the profile verification date.",
                )
            )

    for rule in profile.rules:
        for source_id in rule.source_ids:
            if source_id not in source_ids:
                issues.append(
                    GovernanceIssue(
                        "rule.unknown_source",
                        f"Rule {rule.id!r} cites unknown source {source_id!r}.",
                    )
                )
        try:
            validate_probe_constraint(rule.probe, rule.constraint, rule.phases, label=rule.id)
        except ValueError as exc:
            issues.append(GovernanceIssue("rule.probe", str(exc)))
        if rule.level is RuleLevel.REQUIRED and rule.verification is VerificationMode.UNSUPPORTED:
            issues.append(
                GovernanceIssue(
                    "rule.unsupported_required",
                    f"Required rule {rule.id!r} cannot be verified automatically or manually.",
                    False,
                )
            )
    return GovernanceReport(tuple(issues))


def require_governance(profile: VenueProfile) -> None:
    """Raise a compact error when governance validation fails."""

    report = validate_profile_governance(profile)
    if not report.passed:
        detail = "; ".join(issue.message for issue in report.errors)
        raise ValueError(f"Profile governance validation failed: {detail}")
