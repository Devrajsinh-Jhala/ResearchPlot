from __future__ import annotations

from dataclasses import replace

import pytest

import researchplot as rp
from researchplot.compliance import _evaluate_constraint
from researchplot.governance import require_governance, validate_profile_governance
from researchplot.models import ProfileGovernance, ProfileStatus


@pytest.mark.parametrize(
    ("operator", "expected", "observed", "result"),
    [
        (rp.ConstraintOperator.EQ, "pdf", "pdf", True),
        (rp.ConstraintOperator.NE, "eps", "pdf", True),
        (rp.ConstraintOperator.GT, 4, 5, True),
        (rp.ConstraintOperator.GTE, 5, 5, True),
        (rp.ConstraintOperator.LT, 6, 5, True),
        (rp.ConstraintOperator.LTE, 5, 5, True),
        (rp.ConstraintOperator.IN, ("pdf", "eps"), "pdf", True),
        (rp.ConstraintOperator.NOT_IN, ("png", "jpeg"), "pdf", True),
        (rp.ConstraintOperator.CONTAINS, "embedded", ("subset", "embedded"), True),
        (rp.ConstraintOperator.NOT_CONTAINS, "script", ("font", "image"), True),
        (rp.ConstraintOperator.SUBSET, ("RGB", "CMYK"), ("RGB",), True),
        (rp.ConstraintOperator.BETWEEN, (4, 8), 6, True),
        (rp.ConstraintOperator.APPROX, 89.0, 89.4, True),
        (rp.ConstraintOperator.REQUIRED, True, "caption", True),
        (rp.ConstraintOperator.PROHIBITED, True, False, True),
        (rp.ConstraintOperator.PROHIBITED, "JavaScript", "none", True),
    ],
)
def test_v1_compatibility_constraint_matrix(
    operator: rp.ConstraintOperator,
    expected: object,
    observed: object,
    result: bool,
) -> None:
    tolerance = 0.5 if operator is rp.ConstraintOperator.APPROX else None
    constraint = rp.RuleConstraint(operator, expected, tolerance=tolerance)  # type: ignore[arg-type]
    assert _evaluate_constraint(observed, constraint) is result


@pytest.mark.parametrize(
    ("constraint", "observed", "message"),
    [
        (rp.RuleConstraint(rp.ConstraintOperator.GT, "large"), 4, "expected value"),
        (rp.RuleConstraint(rp.ConstraintOperator.IN, 1), "pdf", "expected value"),
        (rp.RuleConstraint(rp.ConstraintOperator.CONTAINS, "pdf"), 1, "observed value"),
        (rp.RuleConstraint(rp.ConstraintOperator.SUBSET, ("pdf",)), 1, "observed value"),
        (rp.RuleConstraint(rp.ConstraintOperator.BETWEEN, (1,)), 1, "exactly two"),
        (rp.RuleConstraint(rp.ConstraintOperator.BETWEEN, ("low", 2)), 1, "lower bound"),
        (rp.RuleConstraint(rp.ConstraintOperator.APPROX, 1), "one", "observed value"),
    ],
)
def test_v1_compatibility_constraints_fail_closed(
    constraint: rp.RuleConstraint, observed: object, message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        _evaluate_constraint(observed, constraint)


def test_profile_governance_reports_errors_warnings_and_requirement() -> None:
    profile = rp.resolve_profile("nature@2026.08.0")
    source = replace(
        profile.sources[0],
        content_sha256="invalid",
        verified_on="2026-08-14",
    )
    unknown_rule = replace(
        profile.rules[0],
        probe="unknown.probe",
        source_ids=("missing-source",),
    )
    unsupported_rule = replace(
        profile.rules[1],
        verification=rp.VerificationMode.UNSUPPORTED,
        level=rp.RuleLevel.REQUIRED,
    )
    invalid = replace(
        profile,
        sources=(source,),
        rules=(unknown_rule, unsupported_rule),
        maintainers=(),
        governance=ProfileGovernance(),
    )

    report = validate_profile_governance(invalid)
    codes = {issue.code for issue in report.issues}
    assert {
        "governance.maintainers",
        "governance.reviewers",
        "governance.reviewed_on",
        "source.digest",
        "source.verified_after_profile",
        "rule.unknown_source",
        "rule.probe",
        "rule.unsupported_required",
    } <= codes
    assert report.passed is False
    assert report.errors and report.warnings
    with pytest.raises(ValueError, match="governance validation failed"):
        require_governance(invalid)

    translated = validate_profile_governance(replace(profile, status=ProfileStatus.TRANSLATED))
    assert translated.passed is True
    assert translated.warnings[0].code == "governance.translated"
